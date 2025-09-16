from collections import deque
from dataclasses import dataclass, field
import inspect
import math
import os
from typing import Any, Deque, Dict, Iterable, List, Mapping, Optional, Tuple, Union

from omegaconf import DictConfig, OmegaConf
import torch


def encode_batch(tokenizer: Any, texts: List[str]) -> List[List[int]]:
    tok = getattr(tokenizer, "tokenizer", None) or tokenizer
    enc = tok(texts, add_special_tokens=False, padding=False, return_attention_mask=False)  # type: ignore[misc]
    ids = enc["input_ids"] if isinstance(enc, dict) else getattr(enc, "input_ids", enc)
    return ids


class StepGeneration:
    def __init__(
        self,
        max_steps: int,
        *,
        step_token: Optional[str] = None,
        tokens_per_step: Optional[int] = None,
        stop_token: Optional[str] = None,
        include_stop_str_in_output: bool = True,
    ) -> None:
        # exactly one of step_token or tokens_per_step
        if (step_token is None) == (tokens_per_step is None):
            raise ValueError("Provide exactly one of step_token or tokens_per_step")
        if tokens_per_step is not None and tokens_per_step <= 0:
            raise ValueError("tokens_per_step must be a positive integer")

        self.max_steps = int(max_steps)
        self.step_token = step_token
        self.tokens_per_step = tokens_per_step
        self.stop_token = stop_token
        self.include_stop_str_in_output = include_stop_str_in_output


class TokenConfidenceTracker:
    def __init__(self, mode: str, window_size: int) -> None:
        self.mode = mode
        self.window = max(int(window_size), 1)
        self.values: Deque[float] = deque(maxlen=self.window)
        self.sum = 0.0
        self.log_sum = 0.0

    def update(self, value: float) -> float:
        val = float(value)
        eps = 1e-12
        if self.mode == "mean":
            if len(self.values) == self.window:
                removed = self.values.popleft()
                self.sum -= removed
            self.values.append(val)
            self.sum += val
            return self.sum / len(self.values)
        else:
            if len(self.values) == self.window:
                removed = self.values.popleft()
                self.log_sum -= math.log(max(removed, eps))
            self.values.append(val)
            self.log_sum += math.log(max(val, eps))
            return math.exp(self.log_sum / len(self.values))

    def clone(self) -> "TokenConfidenceTracker":
        cloned = TokenConfidenceTracker(self.mode, self.window)
        cloned.values = deque(self.values, maxlen=self.window)
        cloned.sum = self.sum
        cloned.log_sum = self.log_sum
        return cloned


@dataclass(slots=True)
class SMCParticle:
    is_stopped: bool
    prompt_text: str
    completion_chunks: List[str]
    total_new: int
    token_tracker: TokenConfidenceTracker
    step_conf_history: List[float]
    step_count: int
    running_sum: float
    running_prod: float
    running_min: float
    last_step_conf: float
    current_conf_value: float
    prm_log_weights: List[float] = field(default_factory=list)

    def update_step_stats(self, value: float) -> None:
        eps = 1e-12
        self.step_conf_history.append(value)
        self.step_count += 1
        self.running_sum += value
        mult = max(value, eps)
        if self.step_count == 1:
            self.running_prod = mult
            self.running_min = value
        else:
            self.running_prod *= mult
            self.running_min = min(self.running_min, value)
        self.last_step_conf = value


def _to_plain_dict(data: Optional[Mapping[str, Any] | Any]) -> Dict[str, Any]:
    if data is None:
        return {}
    if isinstance(data, dict):
        return dict(data)
    if isinstance(data, DictConfig):  # type: ignore[arg-type]
        return OmegaConf.to_container(data, resolve=True)  # type: ignore[no-any-return]
    if hasattr(data, "items"):
        return dict(data.items())

    return dict(data)


class VLLMProcessRewardWrapper:
    def __init__(
        self,
        model_name: str,
        device: str,
        aggregation: str,
        gpu_memory_utilization: float,
    ) -> None:
        from reward_hub.base import AggregationMethod
        from reward_hub.vllm import reward as reward_module

        # reward_hub <-> vLLM compatibility: vLLM >= 0.10 removed the `device`
        # kwarg on LLM. When that happens, strip it before instantiating.
        if not getattr(reward_module, "_smcgrpo_llm_device_patch", False):
            patched = False
            llm_ctor = getattr(reward_module, "LLM", None)
            if llm_ctor is not None:
                try:
                    sig = inspect.signature(llm_ctor.__init__ if isinstance(llm_ctor, type) else llm_ctor)
                except (TypeError, ValueError):
                    sig = None
                if sig is None or "device" not in sig.parameters:
                    original_llm = llm_ctor

                    def _llm_without_device(*args: Any, **kwargs: Any) -> Any:
                        kwargs.pop("device", None)
                        return original_llm(*args, **kwargs)

                    reward_module.LLM = _llm_without_device
                    patched = True
            reward_module._smcgrpo_llm_device_patch = True
            reward_module._smcgrpo_llm_device_patch_applied = patched

        VllmProcessRewardModel = reward_module.VllmProcessRewardModel

        self.aggregation_method = AggregationMethod(aggregation)
        self.separator = "\n\n"
        self.model = VllmProcessRewardModel(
            model_name=model_name,
            device=device,
            gpu_memory_utilization=gpu_memory_utilization,
        )

    def score(self, prompt: str, response_or_responses: Union[str, List[str]]) -> Union[float, List[float]]:
        if isinstance(response_or_responses, str):
            responses = [response_or_responses]
            single = True
        else:
            responses = list(response_or_responses)
            single = False

        messages: List[List[Dict[str, str]]] = []
        for response in responses:
            messages.append(
                [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": response},
                ]
            )

        scores = self.model.score(
            messages=messages,
            aggregation_method=self.aggregation_method,
            step_sep=self.separator,
            return_full_prm_result=False,
        )

        if single:
            return scores[0]
        return scores


def build_prm_model(prm_cfg: Optional[Mapping[str, Any] | Any]) -> Optional[Any]:
    cfg = _to_plain_dict(prm_cfg)
    if not cfg or not bool(cfg.get("use_prm")):
        return None

    model_name = cfg.get("model_name")
    if not model_name:
        raise ValueError("PRM configuration requires 'model_name' when use_prm is True.")

    raw_device = cfg.get("device")
    cuda_idx = cfg.get("cuda")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    visible_list = [d.strip() for d in visible.split(",") if d.strip()] if visible else None

    def _resolve_index(value: Any) -> int:
        try:
            idx = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("prm.cuda must be an integer index") from exc

        if idx < 0:
            raise ValueError("prm.cuda must be non-negative")

        if visible_list:
            if idx < len(visible_list):
                return idx
            str_idx = str(idx)
            if str_idx in visible_list:
                return visible_list.index(str_idx)

        return idx

    if cuda_idx is not None:
        device = f"cuda:{_resolve_index(cuda_idx)}"
    elif isinstance(raw_device, str) and raw_device.startswith("cuda:"):
        suffix = raw_device.split(":", 1)[1]
        if suffix == "":
            device = "cuda"
        else:
            device = f"cuda:{_resolve_index(suffix)}"
    else:
        device = raw_device or "cuda"

    aggregation = str(cfg.get("aggregation", cfg.get("aggregation_method", "model")))
    gpu_mem_util = float(cfg.get("gpu_memory_utilization", 0.8))

    return VLLMProcessRewardWrapper(
        model_name=model_name,
        device=device,
        aggregation=aggregation,
        gpu_memory_utilization=gpu_mem_util,
    )


class SMCVLLM:
    """Self-confidence SMC in vLLM."""

    def __init__(
        self,
        *,
        llm: Any,
        tokenizer: Any,
        num_particles: int,
        pad_token_id: int,
        temperature: float,
        top_p: float,
        top_k: int,
        min_p: float,
        repetition_penalty: float,
        sg: StepGeneration,
        smc_topk: int,
        window_size: int,
        confidence_eta: float,
        max_new_tokens: int,
        scoring: str = "entropy",
        confidence_group: str = "mean",
        confidence_aggregation: str = "current",
        return_all: bool = False,
        prm: Optional[Any] = None,
    ) -> None:
        self.llm = llm
        self.tok = tokenizer
        self.N = int(num_particles)
        self.pad = int(pad_token_id)
        self.temp = float(temperature)
        self.top_p = float(top_p)
        self.top_k = int(top_k)
        self.min_p = float(min_p)
        self.rep = float(repetition_penalty)
        self.sg = sg
        self.smc_topk = int(smc_topk)
        self.window_size = int(window_size)
        self.confidence_eta = float(confidence_eta)
        self.max_new_tokens = int(max_new_tokens)
        self.return_all = bool(return_all)
        self.prm = prm
        self._group_mode = confidence_group
        self._token_conf_fn = self._resolve_token_scorer(scoring)
        self._group_reducer_fn = self._resolve_group_reducer(confidence_group)
        self._step_aggregator_fn = self._resolve_step_aggregator(confidence_aggregation)
        engine = getattr(self.llm, "llm_engine", None)
        mc = getattr(getattr(engine, "vllm_config", None), "model_config", None) or getattr(engine, "model_config", None)
        if mc is not None and self.smc_topk > 0:
            curr = getattr(mc, "max_logprobs", 0)
            if curr != -1 and curr < self.smc_topk:
                mc.max_logprobs = int(self.smc_topk)

        stop_list = [self.sg.step_token] if self.sg.step_token is not None else None
        self._sampling_kwargs = dict(
            n=1,
            temperature=self.temp,
            top_p=self.top_p,
            top_k=self.top_k,
            min_p=self.min_p,
            repetition_penalty=self.rep,
            stop=stop_list,
            detokenize=True,
            logprobs=self.smc_topk if self.smc_topk > 0 else None,
        )

    def _build_sampling_params(self, max_tokens: int) -> Any:
        from vllm import SamplingParams

        return SamplingParams(max_tokens=max_tokens, **self._sampling_kwargs)

    @staticmethod
    def _inv_sigmoid(score: float) -> float:
        eps = 1e-7
        score = float(score)
        if math.isnan(score):
            return 0.0
        score = min(max(score, eps), 1.0 - eps)
        return math.log(score / (1.0 - score))

    @staticmethod
    def _sigmoid(value: float) -> float:
        if math.isnan(value):
            return 0.0
        if value >= 0:
            z = math.exp(-value)
            return 1.0 / (1.0 + z)
        z = math.exp(value)
        return z / (1.0 + z)

    def _entropy_conf_from_logprobs(self, logprob_entry) -> float:
        if logprob_entry is None:
            return 0.0

        if isinstance(logprob_entry, dict):
            vals = [
                float(getattr(v, "logprob", v))
                for v in logprob_entry.values()
                if getattr(v, "logprob", v) is not None
            ]
        else:
            vals = [
                float(getattr(x, "logprob"))
                for x in logprob_entry
                if getattr(x, "logprob", None) is not None
            ]

        if not vals:
            return 0.0

        vals_tensor = torch.tensor(vals, dtype=torch.float32)
        if self.smc_topk > 0 and vals_tensor.numel() > self.smc_topk:
            vals_tensor = torch.topk(vals_tensor, self.smc_topk).values

        max_val = torch.max(vals_tensor)
        probs = torch.exp(vals_tensor - max_val)
        probs_sum = probs.sum()
        if probs_sum <= 0:
            return 0.0
        probs = probs / probs_sum
        entropy = -(probs * torch.log(probs + 1e-20)).sum()
        return torch.exp(-entropy).item()

    @staticmethod
    def _reduce_mean(values: List[float]) -> float:
        return float(sum(values) / len(values))

    @staticmethod
    def _reduce_geo(values: List[float]) -> float:
        eps = 1e-12
        log_sum = 0.0
        for val in values:
            log_sum += math.log(max(val, eps))
        return float(math.exp(log_sum / len(values)))

    @staticmethod
    def _agg_current(particle: SMCParticle) -> float:
        return float(particle.last_step_conf if particle.step_count else 0.0)

    @staticmethod
    def _agg_mean(particle: SMCParticle) -> float:
        return float(particle.running_sum / particle.step_count) if particle.step_count else 0.0

    @staticmethod
    def _agg_prod(particle: SMCParticle) -> float:
        return float(particle.running_prod) if particle.step_count else 0.0

    @staticmethod
    def _agg_min(particle: SMCParticle) -> float:
        return float(particle.running_min) if particle.step_count else 0.0

    def _resolve_token_scorer(self, scoring: str):
        if scoring == "entropy":
            return self._entropy_conf_from_logprobs
        raise ValueError(f"Unsupported confidence.scoring: {scoring}")

    def _resolve_group_reducer(self, mode: str):
        if mode == "mean":
            return self._reduce_mean
        if mode == "geo":
            return self._reduce_geo
        raise ValueError(f"Unsupported confidence.group mode: {mode}")

    def _resolve_step_aggregator(self, mode: str):
        if mode == "current":
            return self._agg_current
        if mode == "mean":
            return self._agg_mean
        if mode == "prod":
            return self._agg_prod
        if mode == "min":
            return self._agg_min
        raise ValueError(f"Unsupported confidence.aggregation mode: {mode}")

    def generate(
        self,
        prompts_text: List[str],
        original_prompts: List[Any],
    ) -> Union[List[List[int]], Tuple[List[List[int]], List[List[Tuple[int, List[int]]]]]]:
        total = len(prompts_text)
        if total % self.N != 0:
            raise ValueError(f"SMC expects G*N items (N={self.N}). Got total={total}.")
        G = total // self.N
        headers = [prompts_text[g * self.N] for g in range(G)]
        particles: List[SMCParticle] = []
        for idx in range(total):
            particles.append(
                SMCParticle(
                    is_stopped=False,
                    prompt_text=headers[idx // self.N],
                    completion_chunks=[],
                    total_new=0,
                    token_tracker=TokenConfidenceTracker(self._group_mode, self.window_size),
                    step_conf_history=[],
                    step_count=0,
                    running_sum=0.0,
                    running_prod=1.0,
                    running_min=1.0,
                    last_step_conf=0.0,
                    current_conf_value=1.0,
                )
            )
        saved_groups: Optional[List[List[Tuple[int, str]]]] = (
            [[] for _ in range(G)] if self.return_all else None
        )
        for _step in range(self.sg.max_steps):
            batch_inputs: List[str] = []
            idx_map: List[int] = []
            sampling_params: List[Any] = []
            for idx, p in enumerate(particles):
                if p.is_stopped:
                    continue

                remaining_tokens: Optional[int] = None
                if self.max_new_tokens > 0:
                    remaining_tokens = self.max_new_tokens - p.total_new
                    if remaining_tokens <= 0:
                        p.is_stopped = True
                        continue

                step_limit = self.sg.tokens_per_step
                if step_limit is not None:
                    max_tokens = step_limit
                    if remaining_tokens is not None:
                        max_tokens = min(max_tokens, remaining_tokens)
                else:
                    max_tokens = remaining_tokens if remaining_tokens is not None else self.max_new_tokens

                if max_tokens is None or max_tokens <= 0:
                    p.is_stopped = True
                    continue

                prompt_text = p.prompt_text
                if isinstance(self.sg.step_token, str) and self.sg.step_token and not prompt_text.endswith(self.sg.step_token):
                    prompt_text = prompt_text + self.sg.step_token

                batch_inputs.append(prompt_text)
                idx_map.append(idx)
                sampling_params.append(self._build_sampling_params(int(max_tokens)))

            if not batch_inputs:
                break

            outs = self.llm.generate(batch_inputs, sampling_params=sampling_params, use_tqdm=True)

            prm_requests: Dict[int, List[Tuple[int, str]]] = {} if self.prm is not None else {}

            for k, out in enumerate(outs):
                idx = idx_map[k]
                p = particles[idx]
                if not out.outputs:
                    p.is_stopped = True
                    continue
                o = out.outputs[0]
                raw_text = getattr(o, "text", "") or ""
                append_text = raw_text
                stop_reason = getattr(o, "stop_reason", None)
                finish_reason = getattr(o, "finish_reason", None)

                step_stop = (
                    isinstance(self.sg.step_token, str)
                    and self.sg.step_token
                    and isinstance(stop_reason, str)
                    and stop_reason == self.sg.step_token
                )

                should_stop = False

                if isinstance(self.sg.stop_token, str) and self.sg.stop_token:
                    idx = raw_text.find(self.sg.stop_token)
                    if idx != -1:
                        should_stop = True
                        if not self.sg.include_stop_str_in_output:
                            append_text = raw_text[:idx]

                if append_text:
                    p.completion_chunks.append(append_text)
                    p.prompt_text = p.prompt_text + append_text

                step_ids = getattr(o, "token_ids", None) or []
                p.total_new += len(step_ids)

                if self.prm is not None:
                    group_idx = idx // self.N
                    prm_requests.setdefault(group_idx, []).append((idx, "".join(p.completion_chunks)))

                token_group_values: List[float] = []
                if self.smc_topk != 0:
                    lp_list = getattr(o, "logprobs", None) or []
                    for lp_entry in lp_list:
                        token_conf = self._token_conf_fn(lp_entry)
                        group_value = p.token_tracker.update(token_conf)
                        token_group_values.append(group_value)

                if token_group_values:
                    step_conf = self._group_reducer_fn(token_group_values)
                    p.update_step_stats(step_conf)
                    p.current_conf_value = self._step_aggregator_fn(p)

                if finish_reason == "stop" and not step_stop:
                    should_stop = True
                elif stop_reason and not step_stop:
                    should_stop = True

                if self.max_new_tokens and p.total_new >= self.max_new_tokens:
                    should_stop = True

                if should_stop:
                    p.is_stopped = True
                elif isinstance(self.sg.step_token, str) and self.sg.step_token:
                    if not p.prompt_text.endswith(self.sg.step_token):
                        p.prompt_text += self.sg.step_token

            if self.prm is not None and prm_requests:
                for group_idx, items in prm_requests.items():
                    prompt = headers[group_idx]
                    responses = [text for _, text in items]
                    scores = self.prm.score(prompt, responses)
                    if isinstance(scores, Iterable) and not isinstance(scores, (str, bytes)):
                        scores_iter = list(scores)
                    else:
                        scores_iter = [scores] * len(items)
                    if len(scores_iter) != len(items):
                        raise RuntimeError("PRM returned a mismatched number of scores.")
                    for (particle_idx, _), score_val in zip(items, scores_iter):
                        if isinstance(score_val, torch.Tensor):
                            score_float = float(score_val.detach().float().item())
                        else:
                            score_float = float(score_val)
                        logit = self._inv_sigmoid(score_float)
                        particles[particle_idx].prm_log_weights.append(logit)

            # Compute particle weights from the sliding confidence window
            if self.prm is not None:
                weights = torch.tensor(
                    [
                        (
                            self._sigmoid(p.prm_log_weights[-1])
                            if p.prm_log_weights
                            else (1.0 if not p.is_stopped else 0.0)
                        )
                        for p in particles
                    ],
                    dtype=torch.float32,
                )
            else:
                weights = torch.tensor(
                    [
                        (
                            (
                                max(p.current_conf_value if p.step_conf_history else 1.0, 1e-12)
                                ** self.confidence_eta
                            )
                            if not p.is_stopped
                            else 0.0
                        )
                        for p in particles
                    ],
                    dtype=torch.float32,
                )

            for g in range(G):
                start = g * self.N
                end = start + self.N
                alive_indices = [j for j in range(self.N) if not particles[start + j].is_stopped]
                
                if len(alive_indices) <= 1:
                    continue
                
                ws = weights[start:end]
                ws_alive = ws[alive_indices].clone()
                
                if ws_alive.sum() <= 0:
                    ws_alive = torch.ones_like(ws_alive) / len(alive_indices)
                else:
                    ws_alive = ws_alive / ws_alive.sum()
                
                sampled = torch.multinomial(ws_alive, num_samples=self.N, replacement=True)
                parents = [alive_indices[idx] for idx in sampled.tolist()]
                
                if saved_groups is not None:
                    selected_counts = {idx_parent: 0 for idx_parent in alive_indices}
                    for idx_parent in parents:
                        selected_counts[idx_parent] = selected_counts.get(idx_parent, 0) + 1
                    for local_idx in alive_indices:
                        if selected_counts.get(local_idx, 0) == 0:
                            src = particles[start + local_idx]
                            saved_groups[g].append((
                                _step,
                                "".join(src.completion_chunks),
                            ))
                
                new_group: List[SMCParticle] = []
                
                for idx_parent in parents:
                    src = particles[start + idx_parent]
                    new_group.append(
                        SMCParticle(
                            is_stopped=src.is_stopped,
                            prompt_text=src.prompt_text,
                            completion_chunks=list(src.completion_chunks),
                            total_new=int(src.total_new),
                            token_tracker=src.token_tracker.clone(),
                            step_conf_history=list(src.step_conf_history),
                            step_count=int(src.step_count),
                            running_sum=float(src.running_sum),
                            running_prod=float(src.running_prod),
                            running_min=float(src.running_min),
                            last_step_conf=float(src.last_step_conf),
                            current_conf_value=float(src.current_conf_value),
                            prm_log_weights=list(src.prm_log_weights),
                        )
                    )
                for j, particle in enumerate(new_group):
                    particles[start + j] = particle

            if all(p.is_stopped for p in particles):
                break

        completions_text: List[str] = []
        for p in particles:
            completions_text.append("".join(p.completion_chunks))
        encoded_final = encode_batch(self.tok, completions_text)
        if not self.return_all or saved_groups is None:
            return encoded_final

        encoded_saved: List[List[Tuple[int, List[int]]]] = []
        for group in saved_groups:
            if not group:
                encoded_saved.append([])
                continue
            steps, texts = zip(*group)
            encoded_texts = encode_batch(self.tok, list(texts))
            encoded_saved.append([
                (step, ids) for step, ids in zip(steps, encoded_texts)
            ])

        return encoded_final, encoded_saved
