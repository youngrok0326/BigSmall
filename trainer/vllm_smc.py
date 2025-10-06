from collections import defaultdict, deque
from dataclasses import dataclass, field
import logging
import math
import multiprocessing as mp
import os
from typing import Any, Deque, Dict, Iterable, List, Mapping, Optional, Tuple, Union

import torch

from trainer.utils import (
    apply_prm_env_flags,
    encode_batch,
    parse_cuda_visible_from_device,
    patch_reward_module,
    resolve_prm_device,
    to_plain_dict,
)

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

    def ensure_step_suffix(self, text: str) -> str:
        if isinstance(self.step_token, str) and self.step_token and not text.endswith(self.step_token):
            return text + self.step_token
        return text

    def split_on_stop_token(self, text: str) -> Tuple[str, bool]:
        if not (self.stop_token and isinstance(self.stop_token, str)):
            return text, False
        stop_pos = text.find(self.stop_token)

        if stop_pos == -1:
            return text, False
        if self.include_stop_str_in_output:
            return text, True
        return text[:stop_pos], True

class TokenConfidenceTracker:

    def __init__(self, mode: str, window_size: int) -> None:
        self.mode = mode
        self._use_geo = mode == "geo"
        self.window = max(int(window_size), 1)
        self.values: Deque[float] = deque(maxlen=self.window)
        self.sum = 0.0
        self.log_sum = 0.0

    def update(self, value: float) -> float:
        val = float(value)
        eps = 1e-12

        if not self._use_geo:
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
    base_prompt_token_len: int
    prompt_token_len: int
    token_tracker: TokenConfidenceTracker
    step_count: int
    running_sum: float
    running_prod: float
    running_min: float
    last_step_conf: float
    current_conf_value: float
    completion_token_ids: List[int] = field(default_factory=list)
    prm_log_weights: List[float] = field(default_factory=list)

    def update_step_stats(self, value: float) -> None:
        eps = 1e-12
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


@dataclass(slots=True)
class _BaseConfidenceRequest:
    particle_idx: int
    prompt_text: str
    start: int
    end: int

def _prm_worker_main(
    req_q: mp.Queue,
    resp_q: mp.Queue,
    model_name: str,
    device: str | int | None,
    visible_override: Optional[str],
    aggregation_method: str,
    gpu_memory_utilization: float,
) -> None:
    try:
        apply_prm_env_flags()
        logging.getLogger("vllm").setLevel(logging.WARNING)
        override = visible_override if visible_override is not None else parse_cuda_visible_from_device(device)

        if override is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = override
        print(f"[PRM worker] request device={device}, override={override}, env={os.environ.get('CUDA_VISIBLE_DEVICES')}")
        from reward_hub.base import AggregationMethod  # type: ignore
        from reward_hub.vllm import reward as reward_module  # type: ignore
        patch_reward_module(reward_module)
        VllmProcessRewardModel = reward_module.VllmProcessRewardModel
        try:
            agg = AggregationMethod(aggregation_method)
        except Exception:
            agg = AggregationMethod.PROD
        try:
            prm = VllmProcessRewardModel(
                model_name=model_name,
                device=device,
                gpu_memory_utilization=gpu_memory_utilization,
            )
        except TypeError as exc:
            if "device" in str(exc):
                logging.getLogger(__name__).warning(
                    "reward_hub VllmProcessRewardModel rejected device argument; retrying without it",
                )
                prm = VllmProcessRewardModel(
                    model_name=model_name,
                    gpu_memory_utilization=gpu_memory_utilization,
                )
            else:
                raise
        resp_q.put(("ready", None))

        while True:
            batch = req_q.get()

            if batch is None:
                break

            messages = [
                [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": response},
                ]
                for prompt, response in batch
            ]
            scores = prm.score(
                messages=messages,
                aggregation_method=agg,
                return_full_prm_result=False,
                use_tqdm=False,
            )
            resp_q.put(scores)
    except Exception as exc:  # pragma: no cover - defensive
        resp_q.put(("error", f"{type(exc).__name__}: {exc}"))

class LocalVLLMProcessRewardModel:
    """Launch PRM inside a dedicated subprocess pinned to a physical GPU."""

    def __init__(
        self,
        *,
        model_name: str,
        device: str,
        visible_override: Optional[str],
        aggregation_method: str,
        gpu_memory_utilization: float,
        startup_timeout_s: float = 300.0,
    ) -> None:
        ctx = mp.get_context("spawn")
        self._req_q: mp.Queue = ctx.Queue()
        self._resp_q: mp.Queue = ctx.Queue()
        self._worker = ctx.Process(
            target=_prm_worker_main,
            args=(
                self._req_q,
                self._resp_q,
                model_name,
                device,
                visible_override,
                aggregation_method,
                gpu_memory_utilization,
            ),
            daemon=True,
        )

        env_override = visible_override
        prev_visible = None

        if env_override is not None:
            prev_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
            os.environ["CUDA_VISIBLE_DEVICES"] = env_override
        try:
            self._worker.start()
        finally:
            if env_override is not None:
                if prev_visible is None:
                    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
                else:
                    os.environ["CUDA_VISIBLE_DEVICES"] = prev_visible
        try:
            msg, payload = self._resp_q.get(timeout=startup_timeout_s)
        except Exception as exc:
            self.close()
            raise RuntimeError("Timed out while starting PRM subprocess") from exc
        if msg == "error":
            self.close()
            raise RuntimeError(f"Failed to initialize PRM worker: {payload}")

    def close(self) -> None:
        try:
            self._req_q.put(None)
        except Exception:
            pass
        try:
            if self._worker.is_alive():
                self._worker.join(timeout=2.0)
        except Exception:
            pass

    def __del__(self) -> None:  # pragma: no cover
        self.close()

    def score(self, prompt: str, responses: List[str] | str) -> List[float] | float:
        single = isinstance(responses, str)
        batch = [(prompt, r) for r in ([responses] if single else list(responses))]
        scores = self.score_batch(batch)
        return scores[0] if single else scores

    def score_batch(self, prompt_response_pairs: List[Tuple[str, str]]) -> List[float]:
        if not prompt_response_pairs:
            return []
        self._req_q.put(prompt_response_pairs)
        payload = self._resp_q.get()

        if isinstance(payload, tuple) and payload and payload[0] == "error":
            raise RuntimeError(f"PRM worker error: {payload[1]}")
        if not isinstance(payload, list):
            raise RuntimeError("PRM worker returned unexpected payload")
        return payload

def build_prm_model(prm_cfg: Optional[Mapping[str, Any] | Any]) -> Optional[Any]:
    cfg = to_plain_dict(prm_cfg)

    if not cfg or not bool(cfg.get("use_prm")):
        return None
    model_name = cfg.get("model_name")

    if not model_name:
        raise ValueError("PRM configuration requires 'model_name' when use_prm is True.")
    device, override = resolve_prm_device(cfg)
    aggregation = str(cfg.get("aggregation", cfg.get("aggregation_method", "model")))
    gpu_mem_util = float(cfg.get("gpu_memory_utilization", 0.8))
    startup_timeout = float(cfg.get("startup_timeout_s", 300.0))
    return LocalVLLMProcessRewardModel(
        model_name=model_name,
        device=device,
        visible_override=override,
        aggregation_method=aggregation,
        gpu_memory_utilization=gpu_mem_util,
        startup_timeout_s=startup_timeout,
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
        cdf_alpha: float = 0.25,
        max_new_tokens: int,
        max_model_len: Optional[int] = None,
        scoring: str = "entropy",
        confidence_group: str = "mean",
        confidence_aggregation: str = "last",
        return_all: bool = False,
        return_eos: bool = False,
        wandb_logging: bool = False,
        prm: Optional[Any] = None,
        random_sampling: bool = False,
        confidence_from_base: bool = False,
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
        raw_topk = int(smc_topk)
        self.smc_topk = raw_topk
        self.window_size = int(window_size)
        self.confidence_eta = float(confidence_eta)
        self.cdf_alpha = max(float(cdf_alpha), 1e-8)
        self.max_new_tokens = int(max_new_tokens)
        self.max_model_len = int(max_model_len) if max_model_len is not None else None
        self.return_all = bool(return_all)
        self.return_eos = bool(return_eos)
        self.log_wandb = bool(wandb_logging)
        self._call_index = 0
        self.prm = prm
        self._use_prm = prm is not None
        self.random_sampling = bool(random_sampling)
        self._confidence_from_base = bool(confidence_from_base) and not self._use_prm
        self._confidence_enabled = not self._use_prm
        self._base_prompt_logprob_k = max(raw_topk, 1)
        if isinstance(self.sg.step_token, str) and self.sg.step_token:
            encoded_step = self.tok(
                self.sg.step_token,
                add_special_tokens=False,
            )
            step_ids = getattr(encoded_step, "input_ids", None)

            if step_ids is None and isinstance(encoded_step, dict):
                step_ids = encoded_step.get("input_ids")
            if isinstance(step_ids, list) and step_ids and isinstance(step_ids[0], list):
                step_ids = step_ids[0]
            self._step_token_length = len(step_ids or [])
        else:
            self._step_token_length = 0

        scoring = str(scoring).lower()
        confidence_group = str(confidence_group).lower()
        confidence_aggregation = str(confidence_aggregation).lower()
        self._scoring_mode = scoring

        if self._use_prm:
            self.smc_topk = 0
        self._group_mode = confidence_group
        self._token_conf_fn = self._resolve_token_scorer(scoring)
        self._group_reducer_fn = self._resolve_group_reducer(confidence_group)
        self._step_aggregator_fn = self._resolve_step_aggregator(confidence_aggregation)
        engine = getattr(self.llm, "llm_engine", None)
        mc = getattr(getattr(engine, "vllm_config", None), "model_config", None) or getattr(engine, "model_config", None)

        if mc is not None:
            if self._use_prm:
                if getattr(mc, "max_logprobs", None) != 0:
                    mc.max_logprobs = 0
            elif self.smc_topk > 0:
                curr = getattr(mc, "max_logprobs", 0)

                if curr != -1 and curr < self.smc_topk:
                    mc.max_logprobs = int(self.smc_topk)
        stop_list = [self.sg.step_token] if self.sg.step_token is not None else None
        self._request_logprobs = (self.smc_topk > 0) and not self._confidence_from_base and self._confidence_enabled
        self._sampling_kwargs = dict(
            n=1,
            temperature=self.temp,
            top_p=self.top_p,
            top_k=self.top_k,
            min_p=self.min_p,
            repetition_penalty=self.rep,
            stop=stop_list,
            detokenize=True,
            logprobs=self.smc_topk if self._request_logprobs else None,
        )

    def _build_sampling_params(self, max_tokens: int) -> Any:
        from vllm import SamplingParams
        return SamplingParams(max_tokens=max_tokens, **self._sampling_kwargs)

    def _build_base_conf_sampling_params(self) -> Any:
        from vllm import SamplingParams
        return SamplingParams(
            max_tokens=1,
            n=1,
            temperature=1.0,
            top_p=1.0,
            top_k=-1,
            min_p=0.0,
            repetition_penalty=1.0,
            detokenize=False,
            prompt_logprobs=self._base_prompt_logprob_k,
        )

    def _resolve_headers(
        self,
        prompts_text: List[str],
        original_prompts: List[Any],
        groups: int,
    ) -> List[Any]:
        if groups <= 0:
            return []
        total = len(prompts_text)
        # Prefer non-duplicated original prompts when available.

        if isinstance(original_prompts, list):
            if len(original_prompts) == groups:
                return list(original_prompts)
            if len(original_prompts) == total:
                base = [original_prompts[g * self.N] for g in range(groups) if g * self.N < len(original_prompts)]

                if len(base) == groups:
                    return base
        base = [prompts_text[g * self.N] for g in range(groups) if g * self.N < len(prompts_text)]

        if len(base) != groups:
            raise ValueError(
                f"SMC header resolution failed: expected {groups} groups, got {len(base)}"
            )
        return base

    def _group_prompt(
        self,
        group_idx: int,
        headers: List[Any],
        prompts_text: List[str],
        original_prompts: List[Any],
    ) -> Any:
        if 0 <= group_idx < len(headers):
            return headers[group_idx]
        total = len(prompts_text)
        base_idx = group_idx * self.N

        if isinstance(original_prompts, list):
            if len(original_prompts) == total and base_idx < len(original_prompts):
                return original_prompts[base_idx]
            if group_idx < len(original_prompts):
                return original_prompts[group_idx]
        if base_idx < total:
            return prompts_text[base_idx]
        if group_idx < len(prompts_text):
            return prompts_text[group_idx]
        # Fallback to last known header if available to avoid crashes.

        if headers:
            return headers[-1]
        raise IndexError(f"Unable to resolve prompt for group {group_idx}")
    
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

    def _collect_logprob_values(self, logprob_entry) -> List[float]:
        if logprob_entry is None:
            return []
        values: List[float] = []

        if isinstance(logprob_entry, dict):
            for entry in logprob_entry.values():
                val = getattr(entry, "logprob", entry)

                if val is not None:
                    values.append(float(val))
        else:
            for entry in logprob_entry:
                val = getattr(entry, "logprob", None)

                if val is not None:
                    values.append(float(val))
        return values

    def _apply_topk(self, values: torch.Tensor) -> torch.Tensor:
        if self.smc_topk > 0 and values.numel() > self.smc_topk:
            return torch.topk(values, self.smc_topk).values
        return values

    def _entropy_conf_from_logprobs(self, logprob_entry) -> float:
        vals = self._collect_logprob_values(logprob_entry)

        if not vals:
            return 0.0
        vals_tensor = torch.tensor(vals, dtype=torch.float32)

        vals_tensor = self._apply_topk(vals_tensor)
        max_val = torch.max(vals_tensor)
        probs = torch.exp(vals_tensor - max_val)
        probs_sum = probs.sum()

        if probs_sum <= 0:
            return 0.0
        probs = probs / probs_sum
        entropy = -(probs * torch.log(probs + 1e-20)).sum()
        return torch.exp(-entropy).item()

    def _mean_logprob_from_logprobs(self, logprob_entry) -> float:
        vals = self._collect_logprob_values(logprob_entry)

        if not vals:
            return 0.0
        vals_tensor = torch.tensor(vals, dtype=torch.float32)
        vals_tensor = self._apply_topk(vals_tensor)
        mean_logprob = float(vals_tensor.mean().item())
        score = -mean_logprob
        return float(score)

    def _cdf_logprob_from_logprobs(self, logprob_entry) -> float:
        vals = self._collect_logprob_values(logprob_entry)

        if not vals:
            return 0.0
        vals_tensor = torch.tensor(vals, dtype=torch.float32)
        vals_tensor = self._apply_topk(vals_tensor)
        mean_logprob = float(vals_tensor.mean().item())
        x = max(-mean_logprob, 0.0)
        alpha = self.cdf_alpha
        score = 1.0 - math.exp(-alpha * x)
        return float(min(max(score, 1e-6), 0.995))

    def _mean_prob_from_logprobs(self, logprob_entry) -> float:
        vals = self._collect_logprob_values(logprob_entry)

        if not vals:
            return 0.0
        vals_tensor = torch.tensor(vals, dtype=torch.float32)
        vals_tensor = self._apply_topk(vals_tensor)
        probs = torch.exp(vals_tensor)
        mean_prob = float(probs.mean().item())
        return float(max(min(mean_prob, 1.0), 1e-12))
    
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
    def _agg_last(particle: SMCParticle) -> float:
        return float(particle.last_step_conf if particle.step_count else 1.0)
    
    @staticmethod
    def _agg_mean(particle: SMCParticle) -> float:
        return (
            float(particle.running_sum / particle.step_count)

            if particle.step_count
            else 1.0
        )
    
    @staticmethod
    def _agg_prod(particle: SMCParticle) -> float:
        return float(particle.running_prod) if particle.step_count else 1.0
    
    @staticmethod
    def _agg_min(particle: SMCParticle) -> float:
        return float(particle.running_min) if particle.step_count else 1.0

    def _resolve_token_scorer(self, scoring: str):
        if scoring == "entropy":
            return self._entropy_conf_from_logprobs
        if scoring == "logprob":
            return self._mean_logprob_from_logprobs
        if scoring == "prob":
            return self._mean_prob_from_logprobs
        if scoring == "cdf":
            return self._cdf_logprob_from_logprobs
        raise ValueError(f"Unsupported confidence.scoring: {scoring}")

    def _resolve_group_reducer(self, mode: str):
        if mode == "mean":
            return self._reduce_mean
        if mode == "geo":
            return self._reduce_geo
        raise ValueError(f"Unsupported confidence.group mode: {mode}")

    def _resolve_step_aggregator(self, mode: str):
        if mode == "last":
            return self._agg_last
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
        *,
        lora_request: Optional[Any] = None,
    ) -> Union[List[List[int]], Dict[str, Any]]:
        self._call_index += 1
        global_call_step = self._call_index
        total = len(prompts_text)

        if total % self.N != 0:
            raise ValueError(f"SMC expects G*N items (N={self.N}). Got total={total}.")
        G = total // self.N
        headers = self._resolve_headers(prompts_text, original_prompts, G)
        header_token_lens: List[int] = []

        if headers:
            header_token_ids = encode_batch(self.tok, list(headers))  # type: ignore[arg-type]
            header_token_lens = [len(ids) for ids in header_token_ids]

        particles: List[SMCParticle] = [
            SMCParticle(
                is_stopped=False,
                prompt_text=headers[idx // self.N],
                completion_chunks=[],
                total_new=0,
                base_prompt_token_len=(
                    header_token_lens[idx // self.N] if idx // self.N < len(header_token_lens) else 0
                ),
                prompt_token_len=(
                    header_token_lens[idx // self.N] if idx // self.N < len(header_token_lens) else 0
                ),
                token_tracker=TokenConfidenceTracker(self._group_mode, self.window_size),
                step_count=0,
                running_sum=0.0,
                running_prod=1.0,
                running_min=1.0,
                last_step_conf=0.0,
                current_conf_value=1.0,
            )

            for idx in range(total)
        ]
        collect_extras = self.return_all or self.return_eos
        saved_groups: Optional[List[List[Tuple[int, List[int], str, float]]]] = (
            [[] for _ in range(G)] if self.return_all else None
        )
        terminated_records: Optional[List[List[Tuple[int, int, List[int], str, float]]]] = (
            [[] for _ in range(G)] if collect_extras else None
        )

        per_step_not_selected: List[List[int]] = []
        per_step_weight_std: List[List[float]] = []
        per_step_eos: List[List[int]] = []

        for step_idx in range(self.sg.max_steps):
            prev_completion_lens = [len(p.completion_token_ids) for p in particles]
            base_conf_requests: Optional[List[_BaseConfidenceRequest]] = [] if self._confidence_from_base else None
            batch_inputs, idx_map, sampling_params = self._prepare_generation_batch(particles)

            if not batch_inputs:
                break
            outs = self.llm.generate(
                batch_inputs,
                sampling_params=sampling_params,
                use_tqdm=False,
                lora_request=lora_request,
            )
            prm_requests: Dict[int, List[Tuple[int, str]]] = defaultdict(list) if self._use_prm else {}
            step_eos_counter: Optional[List[int]] = [0 for _ in range(G)] if self.log_wandb else None

            for local_idx, out in enumerate(outs):
                particle_idx = idx_map[local_idx]
                self._handle_particle_output(
                    step_idx,
                    particle_idx,
                    particles[particle_idx],
                    out,
                    prev_completion_lens[particle_idx],
                    prm_requests,
                    terminated_records,
                    step_eos_counter,
                    base_conf_requests,
                )
            if self._confidence_from_base and base_conf_requests:
                self._update_confidence_from_base(base_conf_requests, particles)
            if self._use_prm and prm_requests:
                self._apply_prm_scores(prm_requests, particles, headers, prompts_text, original_prompts)
            weights = self._build_particle_weights(particles)
            not_selected_counts, weight_stds = self._resample_particles(
                weights, particles, G, step_idx, saved_groups
            )

            if self.log_wandb:
                per_step_not_selected.append(not_selected_counts)
                per_step_weight_std.append(weight_stds)
                if step_eos_counter is not None:
                    per_step_eos.append(step_eos_counter)
                else:
                    per_step_eos.append([0 for _ in range(G)])

            if all(p.is_stopped for p in particles):
                break
        encoded_final = [list(p.completion_token_ids) for p in particles]

        if self.log_wandb and global_call_step % 10 == 0:
            self._log_resampling_table(
                global_call_step,
                per_step_not_selected,
                per_step_eos,
                per_step_weight_std,
            )

        primary_encoded_by_group: List[List[List[int]]] = []
        primary_conf_by_group: List[List[float]] = []

        for g in range(G):
            start = g * self.N
            end = start + self.N
            primary_encoded_by_group.append([list(ids) for ids in encoded_final[start:end]])
            primary_conf_by_group.append(
                [float(particles[start + idx].current_conf_value) for idx in range(self.N)]
            )

        extras_tokens_by_group: List[List[List[int]]] = [[] for _ in range(G)]
        extras_text_by_group: List[List[str]] = [[] for _ in range(G)]
        extras_conf_by_group: List[List[float]] = [[] for _ in range(G)]

        if terminated_records is not None:
            final_particle_ids = {id(p) for p in particles}

            for g, records in enumerate(terminated_records):
                for particle_id, _, token_ids, text, conf in records:
                    if particle_id in final_particle_ids:
                        continue
                    extras_tokens_by_group[g].append(list(token_ids))
                    extras_text_by_group[g].append(str(text))
                    extras_conf_by_group[g].append(float(conf))

        extras_encoded_by_group: List[List[List[int]]] = [[] for _ in range(G)]

        for g in range(G):
            if extras_tokens_by_group[g]:
                extras_encoded_by_group[g] = [list(ids) for ids in extras_tokens_by_group[g]]

        completions_by_group: List[List[List[int]]] = []
        completion_confidences_by_group: List[List[float]] = []
        completions_text_by_group: List[List[str]] = []

        for g in range(G):
            base = list(primary_encoded_by_group[g])
            base_conf = list(primary_conf_by_group[g])
            start = g * self.N
            base_text = ["".join(particles[start + idx].completion_chunks) for idx in range(self.N)]

            if extras_encoded_by_group[g]:
                base.extend(extras_encoded_by_group[g])
                base_conf.extend(extras_conf_by_group[g])
                base_text.extend(extras_text_by_group[g])
            completions_by_group.append(base)
            completion_confidences_by_group.append(base_conf)
            completions_text_by_group.append(base_text)

        group_sizes = [len(group) for group in completions_by_group]
        if saved_groups is not None:
            encoded_saved, saved_confidences_by_group = self._encode_saved_groups(saved_groups)
        else:
            encoded_saved = [[] for _ in range(G)]
            saved_confidences_by_group = [[] for _ in range(G)]

        if self.log_wandb and global_call_step % 10 == 0:
            try:
                import wandb  # type: ignore
            except ImportError:
                pass
            else:
                if getattr(wandb, "run", None) is not None:
                    columns = [
                        "global_step",
                        "group_id",
                        "completion_idx",
                        "text",
                    ]
                    table = wandb.Table(columns=columns)

                    for group_idx, texts in enumerate(completions_text_by_group):
                        for completion_idx, text in enumerate(texts):
                            table.add_data(global_call_step, group_idx, completion_idx, text)
                    wandb.log({"smc/completions": table}, commit=False)

        return {
            "group_sizes": group_sizes,
            "completions": completions_by_group,
            "completion_confidences": completion_confidences_by_group,
            "saved": encoded_saved,
            "saved_confidences": saved_confidences_by_group,
        }

    def _next_token_budget(self, particle: SMCParticle) -> Optional[int]:
        remaining: Optional[int] = None
        model_remaining: Optional[int] = None

        if self.max_model_len is not None:
            model_remaining = self.max_model_len - particle.prompt_token_len

            if model_remaining <= 0:
                return None

        if self.max_new_tokens > 0:
            generated_tokens = max(particle.prompt_token_len - particle.base_prompt_token_len, 0)
            remaining = self.max_new_tokens - generated_tokens

            if remaining <= 0:
                return None
        step_limit = self.sg.tokens_per_step

        if remaining is not None and model_remaining is not None:
            remaining = min(remaining, model_remaining)
        elif remaining is None and model_remaining is not None:
            remaining = model_remaining

        if step_limit is not None:
            return min(step_limit, remaining) if remaining is not None else step_limit
        return remaining if remaining is not None else self.max_new_tokens

    def _prepare_generation_batch(self, particles: List[SMCParticle]) -> Tuple[List[str], List[int], List[Any]]:
        batch_inputs: List[str] = []
        idx_map: List[int] = []
        sampling_params: List[Any] = []

        for idx, particle in enumerate(particles):
            if particle.is_stopped:
                continue
            ensured_prompt = self.sg.ensure_step_suffix(particle.prompt_text)

            if ensured_prompt != particle.prompt_text:
                particle.prompt_text = ensured_prompt
                # Step suffix contributes to the vLLM context, so refresh token accounting immediately.
                self._sync_particle_tokens(particle)
            budget = self._next_token_budget(particle)

            if budget is None or budget <= 0:
                particle.is_stopped = True
                continue
            batch_inputs.append(particle.prompt_text)
            idx_map.append(idx)
            sampling_params.append(self._build_sampling_params(int(budget)))
        return batch_inputs, idx_map, sampling_params

    def _handle_particle_output(
        self,
        step_idx: int,
        particle_idx: int,
        particle: SMCParticle,
        output: Any,
        prev_completion_len: int,
        prm_requests: Dict[int, List[Tuple[int, str]]],
        terminated_records: Optional[List[List[Tuple[int, int, List[int], str, float]]]],
        step_eos_counter: Optional[List[int]],
        base_conf_requests: Optional[List[_BaseConfidenceRequest]],
    ) -> None:
        if not getattr(output, "outputs", None):
            particle.is_stopped = True
            return
        decoded = output.outputs[0]
        raw_text = getattr(decoded, "text", "") or ""
        append_text, explicit_stop = self.sg.split_on_stop_token(raw_text)

        if append_text:
            particle.completion_chunks.append(append_text)
            particle.prompt_text += append_text
        self._sync_particle_tokens(particle)
        group_idx = particle_idx // self.N

        if self._use_prm:
            prm_requests[group_idx].append((particle_idx, "".join(particle.completion_chunks)))
        if self._request_logprobs:
            logprob_entries = getattr(decoded, "logprobs", None) or []
            token_group_values = [
                particle.token_tracker.update(self._token_conf_fn(entry))

                for entry in logprob_entries
            ]

            if token_group_values:
                step_conf = self._group_reducer_fn(token_group_values)
                particle.update_step_stats(step_conf)
                particle.current_conf_value = self._step_aggregator_fn(particle)
        elif base_conf_requests is not None and self._confidence_from_base:
            new_completion_len = len(particle.completion_token_ids)
            if new_completion_len > prev_completion_len:
                start_idx = particle.base_prompt_token_len + prev_completion_len
                end_idx = particle.base_prompt_token_len + new_completion_len
                if end_idx > start_idx:
                    base_conf_requests.append(
                        _BaseConfidenceRequest(
                            particle_idx=particle_idx,
                            prompt_text=particle.prompt_text,
                            start=start_idx,
                            end=end_idx,
                        )
                    )
        stop_reason = getattr(decoded, "stop_reason", None)
        finish_reason = getattr(decoded, "finish_reason", None)
        step_stop = (
            isinstance(self.sg.step_token, str)
            and self.sg.step_token
            and isinstance(stop_reason, str)
            and stop_reason == self.sg.step_token
        )
        should_stop = explicit_stop

        stop_due_to_eos = False

        if finish_reason == "stop" and not step_stop:
            should_stop = True
            stop_due_to_eos = True
        elif stop_reason and not step_stop:
            should_stop = True
        if self.max_new_tokens and particle.total_new >= self.max_new_tokens:
            should_stop = True
        if should_stop:
            particle.is_stopped = True

            if self._use_prm:
                prm_requests[group_idx].append((particle_idx, "".join(particle.completion_chunks)))
            if terminated_records is not None:
                completion_text = "".join(particle.completion_chunks)
                terminated_records[group_idx].append(
                    (
                        id(particle),
                        int(step_idx),
                        list(particle.completion_token_ids),
                        completion_text,
                        float(particle.current_conf_value),
                    )
                )
            if step_eos_counter is not None and stop_due_to_eos:
                step_eos_counter[group_idx] += 1
        else:
            previous_prompt = particle.prompt_text
            updated_prompt = self.sg.ensure_step_suffix(previous_prompt)
            particle.prompt_text = updated_prompt

            if updated_prompt != previous_prompt:
                self._sync_particle_tokens(particle)

    def _apply_prm_scores(
        self,
        prm_requests: Dict[int, List[Tuple[int, str]]],
        particles: List[SMCParticle],
        headers: List[Any],
        prompts_text: List[str],
        original_prompts: List[Any],
    ) -> None:
        if not self.prm:
            return
        batch_entries: List[Tuple[int, str, str]] = []

        for group_idx, items in prm_requests.items():
            prompt = self._group_prompt(group_idx, headers, prompts_text, original_prompts)
            for particle_idx, text in items:
                batch_entries.append((particle_idx, prompt, text))

        if not batch_entries:
            return

        scores = self.prm.score_batch([(prompt, text) for _, prompt, text in batch_entries])

        if len(scores) != len(batch_entries):
            raise RuntimeError("PRM returned a mismatched number of scores.")

        for (particle_idx, _, _), score_val in zip(batch_entries, scores):
            if isinstance(score_val, torch.Tensor):
                score_float = float(score_val.detach().float().item())
            else:
                score_float = float(score_val)
            particles[particle_idx].prm_log_weights.append(self._inv_sigmoid(score_float))

    def _update_confidence_from_base(
        self,
        requests: List[_BaseConfidenceRequest],
        particles: List[SMCParticle],
    ) -> None:
        if not requests:
            return
        sampling_params = self._build_base_conf_sampling_params()
        prompts = [req.prompt_text for req in requests]
        outs = self.llm.generate(
            prompts,
            sampling_params=sampling_params,
            use_tqdm=False,
            lora_request=None,
        )

        for req, out in zip(requests, outs):
            prompt_logprobs = getattr(out, "prompt_logprobs", None)

            if not prompt_logprobs:
                continue
            total_entries = len(prompt_logprobs)
            start = min(max(req.start, 0), total_entries)
            end = min(max(req.end, start), total_entries)

            if start >= end:
                continue
            segment = prompt_logprobs[start:end]
            if not segment:
                continue
            particle = particles[req.particle_idx]
            token_group_values = [
                particle.token_tracker.update(self._token_conf_fn(entry))
                for entry in segment
                if entry is not None
            ]

            if token_group_values:
                step_conf = self._group_reducer_fn(token_group_values)
                particle.update_step_stats(step_conf)
                particle.current_conf_value = self._step_aggregator_fn(particle)

    def _sync_particle_tokens(self, particle: SMCParticle) -> None:
        text = "".join(particle.completion_chunks)
        if text:
            particle.completion_token_ids = encode_batch(self.tok, [text])[0]
        else:
            particle.completion_token_ids = []
        particle.total_new = len(particle.completion_token_ids)
        full_prompt_ids = encode_batch(self.tok, [particle.prompt_text])[0]
        particle.prompt_token_len = len(full_prompt_ids)

    def _build_particle_weights(self, particles: List[SMCParticle]) -> torch.Tensor:
        if self._use_prm:
            values = [
                (math.exp(p.prm_log_weights[-1]) if p.prm_log_weights else (1.0 if not p.is_stopped else 0.0))

                for p in particles
            ]
        else:
            values = [
                (max(p.current_conf_value, 1e-12) ** self.confidence_eta) if not p.is_stopped else 0.0

                for p in particles
            ]
        return torch.tensor(values, dtype=torch.float32)

    def _resample_particles(
        self,
        weights: torch.Tensor,
        particles: List[SMCParticle],
        groups: int,
        step_idx: int,
        saved_groups: Optional[List[List[Tuple[int, List[int], str, float]]]],
    ) -> Tuple[List[int], List[float]]:
        not_selected_counts: List[int] = [0 for _ in range(groups)]
        weight_stds: List[float] = [0.0 for _ in range(groups)]

        for g in range(groups):
            start = g * self.N
            end = start + self.N
            alive = [j for j in range(self.N) if not particles[start + j].is_stopped]

            if len(alive) <= 1:
                not_selected_counts[g] = 0
                weight_stds[g] = 0.0
                continue
            ws_alive = weights[start:end][alive]

            if ws_alive.sum() <= 0:
                ws_alive = torch.ones_like(ws_alive) / len(alive)
            else:
                ws_alive = ws_alive / ws_alive.sum()
            if ws_alive.numel() > 0:
                weight_stds[g] = float(ws_alive.std(unbiased=False).item())
            else:
                weight_stds[g] = 0.0
            if self.random_sampling:
                dead = [j for j in range(self.N) if particles[start + j].is_stopped]

                if not dead:
                    not_selected_counts[g] = 0
                    continue
                sampled = torch.randint(
                    len(alive),
                    (len(dead),),
                    device=ws_alive.device,
                )
                sampled_indices = sampled.cpu().tolist()
                parents = [alive[idx] for idx in sampled_indices]
                not_selected_counts[g] = 0

                for dest_offset, parent in zip(dead, parents):
                    particles[start + dest_offset] = self._clone_particle(
                        particles[start + parent]
                    )
                continue

            sampled = torch.multinomial(ws_alive, num_samples=self.N, replacement=True)
            sampled_indices = sampled.cpu().tolist()
            parents = [alive[idx] for idx in sampled_indices]

            selected_counts = {idx_parent: 0 for idx_parent in alive}

            for idx_parent in parents:
                selected_counts[idx_parent] = selected_counts.get(idx_parent, 0) + 1
            not_sampled = [local_idx for local_idx in alive if selected_counts.get(local_idx, 0) == 0]
            not_selected_counts[g] = len(not_sampled)

            if saved_groups is not None:
                for local_idx in not_sampled:
                    src = particles[start + local_idx]
                    completion_text = "".join(src.completion_chunks)
                    saved_groups[g].append(
                        (
                            step_idx,
                            list(src.completion_token_ids),
                            completion_text,
                            float(src.current_conf_value),
                        )
                    )
            cloned = [self._clone_particle(particles[start + parent]) for parent in parents]

            for offset, particle in enumerate(cloned):
                particles[start + offset] = particle

        return not_selected_counts, weight_stds

    @staticmethod

    def _clone_particle(src: SMCParticle) -> SMCParticle:
        return SMCParticle(
            is_stopped=src.is_stopped,
            prompt_text=src.prompt_text,
            completion_chunks=list(src.completion_chunks),
            total_new=int(src.total_new),
            base_prompt_token_len=int(src.base_prompt_token_len),
            prompt_token_len=int(src.prompt_token_len),
            token_tracker=src.token_tracker.clone(),
            step_count=int(src.step_count),
            running_sum=float(src.running_sum),
            running_prod=float(src.running_prod),
            running_min=float(src.running_min),
            last_step_conf=float(src.last_step_conf),
            current_conf_value=float(src.current_conf_value),
            completion_token_ids=list(src.completion_token_ids),
            prm_log_weights=list(src.prm_log_weights),
        )

    def _encode_saved_groups(
        self, saved_groups: List[List[Tuple[int, List[int], str, float]]]
    ) -> Tuple[List[List[Tuple[int, List[int]]]], List[List[float]]]:
        encoded: List[List[Tuple[int, List[int]]]] = []
        confidences: List[List[float]] = []

        for group in saved_groups:
            if not group:
                encoded.append([])
                confidences.append([])
                continue
            encoded.append([(step, list(ids)) for step, ids, _, _ in group])
            confidences.append([float(conf) for _, _, _, conf in group])
        return encoded, confidences

    def _log_resampling_table(
        self,
        global_step: int,
        per_step_not_selected: List[List[int]],
        per_step_eos: List[List[int]],
        per_step_weight_std: List[List[float]],
    ) -> None:
        if not self.log_wandb or not per_step_not_selected:
            return
        try:
            import wandb  # type: ignore
        except ImportError:
            return
        if getattr(wandb, "run", None) is None:
            return
        columns = [
            "global_step",
            "reasoning_step",
            "mean_not_resampled",
            "mean_eos",
            "mean_weight_std",
        ]
        table = wandb.Table(columns=columns)
        total_steps = len(per_step_not_selected)

        for idx in range(total_steps):
            counts = per_step_not_selected[idx]
            eos_counts = per_step_eos[idx] if idx < len(per_step_eos) else []
            weight_vals = per_step_weight_std[idx] if idx < len(per_step_weight_std) else []
            mean_not = (sum(counts) / len(counts)) if counts else 0.0
            mean_eos = (sum(eos_counts) / len(eos_counts)) if eos_counts else 0.0
            mean_std = (sum(weight_vals) / len(weight_vals)) if weight_vals else 0.0
            table.add_data(global_step, idx + 1, mean_not, mean_eos, mean_std)
        wandb.log({"smc/reasoning_step_metrics": table}, commit=True)
