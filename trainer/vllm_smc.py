import math
import warnings
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence

try:
    from vllm import LLM, SamplingParams
    _VLLM_AVAILABLE = True
except Exception:
    _VLLM_AVAILABLE = False

try:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    _HF_AVAILABLE = True
except Exception:
    _HF_AVAILABLE = False


def _inv_sigmoid(x: float) -> float:
    x = min(max(float(x), 1e-7), 1.0 - 1e-7)
    return math.log(x / (1.0 - x))


def _softmax(log_weights: Sequence[float]) -> List[float]:
    if len(log_weights) == 0:
        return []
    m = max(log_weights)
    exps = [math.exp(lw - m) for lw in log_weights]
    s = sum(exps)
    return [e / s for e in exps]


@dataclass
class Particle:
    steps: List[str]
    is_stopped: bool
    partial_log_weights: List[float]

    @property
    def log_weight(self) -> float:
        if self.partial_log_weights:
            return self.partial_log_weights[-1]
        return 0.0


class StepGeneration:
    def __init__(
        self,
        max_steps: int,
        *,
        step_token: Optional[str] = None,
        tokens_per_step: Optional[int] = None,
        stop_token: Optional[str] = None,
        include_stop_str_in_output: bool = False,
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

    def _post_process(self, steps: List[str], *, stopped: bool) -> str:
        if self.tokens_per_step is not None:
            # chunked by fixed token count: just concat
            resp = "".join(steps)
        else:
            # delimiter-based
            delim = self.step_token or ""
            resp = delim.join(steps)
            if not stopped and isinstance(self.step_token, str):
                resp += self.step_token
        if stopped and isinstance(self.stop_token, str) and not self.include_stop_str_in_output:
            # If we stopped because of stop_token, ensure we don't keep it in the response tail.
            # This is a conservative cleanup; if users want to keep it, set include_stop_str_in_output=True
            resp = resp.replace(self.stop_token, "")
        return resp


class AbstractProcessRewardModel:
    def score(self, prompt: str, response_or_list: str | List[str]) -> float | List[float]:
        raise NotImplementedError


class LocalHFProcessRewardModel(AbstractProcessRewardModel):
    """HF-based PRM scorer: loads a transformers classification model and returns a probability [0,1].

    This is a robust fallback when a vLLM-backed PRM is not available.
    """

    def __init__(self, model_name: str, device: str = "cuda:0") -> None:
        if not _HF_AVAILABLE:
            raise ImportError("transformers/torch are required for HF PRM backend")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, trust_remote_code=True)
        self.model.to(device)
        self.model.eval()
        self.device = device

    def _build_text(self, prompt: str, response: str) -> str:
        # Simple, consistent concatenation for classification models
        # If the PRM expects a special template, adapt it here.
        return f"[INST] {prompt} [/INST] {response}"

    def score(self, prompt: str, response_or_list: str | List[str]) -> float | List[float]:
        single = isinstance(response_or_list, str)
        responses = [response_or_list] if single else response_or_list
        texts = [self._build_text(prompt, r) for r in responses]

        if not _HF_AVAILABLE:
            raise ImportError("HF backend not available for PRM scoring")
        with torch.no_grad():
            enc = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
            out = self.model(**enc)
            logits = out.logits
            # Binary classification assumed; sigmoid over last dim
            probs = (
                torch.sigmoid(logits.squeeze(-1))
                if logits.ndim == 2 and logits.size(-1) == 1
                else torch.softmax(logits, dim=-1)[..., -1]
            )
            probs = probs.detach().float().cpu().tolist()
        return probs[0] if single else probs


class LocalVLLMProcessRewardModel(AbstractProcessRewardModel):
    """vLLM-based PRM wrapper.

    If reward_hub is available, defer to its VllmProcessRewardModel for accurate scoring.
    Otherwise, fall back to an internal HF classifier to keep things self-contained.
    """

    def __init__(
        self,
        *,
        model_name: str,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.25,
        device: str = "cuda:0",
        aggregation_method: str = "prod",
    ) -> None:
        self._backend = None
        self._mode = None

        # Try reward_hub's vLLM PRM first (best parity with its_hub)
        try:
            from reward_hub.base import AggregationMethod  # type: ignore
            from reward_hub.vllm.reward import VllmProcessRewardModel  # type: ignore

            class _RewardHubAdapter:
                def __init__(self, model_name: str, device: str, agg: str):
                    self.model = VllmProcessRewardModel(model_name=model_name, device=device)
                    try:
                        self.agg = AggregationMethod(agg)
                    except Exception:
                        self.agg = AggregationMethod.PROD

                def score(self, prompt: str, responses: List[str]) -> List[float]:
                    messages = [
                        [{"role": "user", "content": prompt}, {"role": "assistant", "content": r}] for r in responses
                    ]
                    return self.model.score(
                        messages=messages,
                        aggregation_method=self.agg,
                        return_full_prm_result=False,
                    )

            self._backend = _RewardHubAdapter(model_name=model_name, device=device, agg=aggregation_method)
            self._mode = "reward_hub_vllm"
            return
        except Exception as e:
            raise ImportError(
                "reward_hub not available for PRM vLLM backend. Install reward_hub to use LocalVLLMProcessRewardModel."
            ) from e

    def score(self, prompt: str, response_or_list: str | List[str]) -> float | List[float]:
        single = isinstance(response_or_list, str)
        responses = [response_or_list] if single else response_or_list
        probs = self._backend.score(prompt, responses)
        return probs[0] if single else probs


class ParticleFilteringVLLM:
    """Particle Filtering over a colocated vLLM policy engine, with per-step PRM scoring.

    - Resamples every step using softmax over current step log-weights (no ESS threshold).
    - Returns all particles' completions (no final selection) as token id lists per prompt.
    """

    def __init__(
        self,
        *,
        llm: Any,  # vLLM policy engine (from Unsloth/TRL colocation)
        tokenizer: Any,  # policy tokenizer used to tokenize final completions
        num_particles: int,
        eos_token_id: int,
        pad_token_id: int,
        temperature: float,
        top_p: float,
        top_k: int,
        min_p: float,
        repetition_penalty: float,
        sg: StepGeneration,
        prm: AbstractProcessRewardModel,
    ) -> None:
        if not _VLLM_AVAILABLE:
            raise ImportError("vLLM is required for ParticleFilteringVLLM")
        self.llm = llm
        self.tok = tokenizer
        self.N = int(num_particles)
        self.eos = int(eos_token_id)
        self.pad = int(pad_token_id)
        self.temp = float(temperature)
        self.top_p = float(top_p)
        self.top_k = int(top_k)
        self.min_p = float(min_p)
        self.rep = float(repetition_penalty)
        self.sg = sg
        self.prm = prm

    def _sampling_params(self, *, max_tokens: int, stop: Optional[List[str]] = None) -> SamplingParams:
        kwargs = dict(
            n=1,
            max_tokens=max_tokens,
            temperature=self.temp,
            top_p=self.top_p,
            top_k=self.top_k,
            min_p=self.min_p,
            repetition_penalty=self.rep,
            stop=stop,
            detokenize=True,
        )
        return SamplingParams(**kwargs)

    def _extract_user_content(self, original_prompt: Any, fallback: str) -> str:
        # Try structured chat format first
        if isinstance(original_prompt, list):
            for msg in original_prompt:
                if isinstance(msg, dict) and msg.get("role") == "user":
                    content = msg.get("content")
                    return content if isinstance(content, str) else fallback
        # Try to extract from base prompt structure
        if isinstance(original_prompt, str) and "Problem:" in original_prompt:
            try:
                segment = original_prompt.split("Problem:", 1)[1]
                segment = segment.split("Solution:", 1)[0]
                return segment.strip()
            except Exception:
                pass
        return fallback

    def generate(
        self,
        prompts_text: List[str],  # templated strings used for generation
        original_prompts: List[Any],  # raw inputs; used to build PRM messages
        images: Optional[List[Any]] = None,
    ) -> List[List[int]]:
        if images is not None:
            warnings.warn("Images provided to ParticleFilteringVLLM.generate; multimodal PF is not implemented. Ignoring images.")

        B = len(prompts_text)
        # Initialize particles per prompt
        particles: List[List[Particle]] = [[Particle(steps=[], is_stopped=False, partial_log_weights=[]) for _ in range(self.N)] for _ in range(B)]

        # Precompute user-only prompts for PRM
        user_prompts: List[str] = [self._extract_user_content(original_prompts[i], fallback=prompts_text[i]) for i in range(B)]

        for step_idx in range(self.sg.max_steps):
            # Collect alive particle contexts
            batch_inputs: List[str] = []
            index_map: List[tuple[int, int]] = []  # (prompt_idx, particle_idx)
            for i in range(B):
                for j in range(self.N):
                    p = particles[i][j]
                    if p.is_stopped:
                        continue
                    context = prompts_text[i] + self.sg._post_process(p.steps, stopped=False)
                    batch_inputs.append(context)
                    index_map.append((i, j))

            if not batch_inputs:
                break

            # Generate one step per alive particle
            if self.sg.step_token is not None:
                sp = self._sampling_params(max_tokens=128, stop=[self.sg.step_token])
            else:
                sp = self._sampling_params(max_tokens=int(self.sg.tokens_per_step or 16), stop=None)

            outs = self.llm.generate(batch_inputs, sampling_params=sp, use_tqdm=False)

            # Update particles with generated step text
            for k, out in enumerate(outs):
                text = out.outputs[0].text if out.outputs else ""
                i, j = index_map[k]
                p = particles[i][j]
                p.steps.append(text)
                # stop on explicit stop_token
                if isinstance(self.sg.stop_token, str) and (self.sg.stop_token in text):
                    p.is_stopped = True

            # Mark particles stopped if max_steps reached after this iteration
            for i in range(B):
                for j in range(self.N):
                    if len(particles[i][j].steps) >= self.sg.max_steps:
                        particles[i][j].is_stopped = True

            # PRM scoring per prompt
            # Build per-prompt, per-particle partial texts
            for i in range(B):
                partials = [self.sg._post_process(p.steps, stopped=True) for p in particles[i]]
                scores = self.prm.score(user_prompts[i], partials)
                # Ensure list of scores
                if not isinstance(scores, list):
                    scores = [scores for _ in range(self.N)]
                # Update weights
                for j in range(self.N):
                    s = float(scores[j])
                    particles[i][j].partial_log_weights.append(_inv_sigmoid(s))

            # Resample every step (per prompt)
            for i in range(B):
                logw = [p.log_weight for p in particles[i]]
                probs = _softmax(logw)
                # Multinomial resample N indices
                # Use Python's random choices to avoid torch dependency where possible
                import random

                idxs = random.choices(list(range(self.N)), weights=probs, k=self.N)
                new_particles = []
                for idx in idxs:
                    src = particles[i][idx]
                    new_particles.append(
                        Particle(steps=list(src.steps), is_stopped=src.is_stopped, partial_log_weights=list(src.partial_log_weights))
                    )
                particles[i] = new_particles

            # If all particles across all prompts are stopped, exit early
            if all(p.is_stopped for group in particles for p in group):
                break

        # Build final completions and tokenize with policy tokenizer
        results: List[List[int]] = []
        for i in range(B):
            for j in range(self.N):
                completion_text = self.sg._post_process(particles[i][j].steps, stopped=True)
                # Tokenize completion only
                enc = self.tok(text=completion_text, return_tensors="pt", add_special_tokens=False)
                ids = enc["input_ids"][0].tolist()
                results.append(ids)
        return results
