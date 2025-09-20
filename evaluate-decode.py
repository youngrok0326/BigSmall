"""
Compare decoding strategies (default HF generate vs custom SMC generate)
on HF checkpoints without LoRA, across multiple datasets, with W&B logging.

Usage:
  uv run python3 evaluate-decode.py
  # Override config on CLI if needed, eg. change model or batch size
  uv run python3 evaluate-decode.py model.model_name=Qwen/Qwen2.5-3B
"""

import os

os.environ["VLLM_LOGGING_LEVEL"] = "WARNING"

import logging

logging.getLogger("vllm").setLevel(logging.WARNING)
import json
import math
from datetime import datetime
from dataclasses import dataclass
from math import comb
from typing import Any, Dict, List, Optional

# Progress bar
try:
    from tqdm.auto import tqdm
except Exception:  # fallback if tqdm is unavailable
    def tqdm(*args, **kwargs):
        class _Noop:
            def __init__(self, total=None, **_):
                self.total = total
            def update(self, n=1):
                pass
            def close(self):
                pass
            def __enter__(self):
                return self
            def __exit__(self, exc_type, exc, tb):
                self.close()
        # If first arg is iterable, just return it
        if args and hasattr(args[0], "__iter__"):
            return args[0]
        return _Noop(total=kwargs.get("total"))

import hydra
from omegaconf import DictConfig, OmegaConf

import torch
from unsloth import FastLanguageModel
from transformers import GenerationConfig

from utils.data import (
    answer_correct,
    format_correct,
    get_questions,
    set_tokenizer_name,
)


@dataclass(frozen=True)
class PassMetric:
    """Specification for a pass@k-style metric."""

    name: str
    is_proportion: bool
    value: float

    def k_for_count(self, n: int) -> int:
        if n <= 0:
            return 0
        if self.is_proportion:
            k = int(round(n * self.value))
            k = max(1, min(n, k))
        else:
            k = int(self.value)
        return k


def _ensure_pad_token(tokenizer):
    if tokenizer.pad_token is None:
        # Use eos as pad if pad doesn't exist
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})


def _build_gen_config_from_section(tokenizer, section_cfg: DictConfig, max_new_tokens: int) -> GenerationConfig:
    _ensure_pad_token(tokenizer)
    return GenerationConfig(
        do_sample=section_cfg.get("do_sample", False),
        temperature=section_cfg.get("temperature", 1.0),
        top_p=section_cfg.get("top_p", 1.0),
        top_k=section_cfg.get("top_k", 0),
        max_new_tokens=max_new_tokens,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        return_dict_in_generate=False,
    )


def _batch_tokenize(tokenizer, texts, max_length, padding_side: str | None = None, add_special_tokens: bool = False):
    """Tokenize prompts with optional padding-side override and no extra specials."""
    prev_side = getattr(tokenizer, "padding_side", None)
    try:
        if padding_side is not None and prev_side is not None:
            tokenizer.padding_side = padding_side
        return tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
            add_special_tokens=add_special_tokens,
        )
    finally:
        if padding_side is not None and prev_side is not None:
            tokenizer.padding_side = prev_side


def _decode_generated(tokenizer, sequences: torch.Tensor, prompt_lens: torch.Tensor):
    """Optimized decoding using a single batch_decode call."""
    tokens_to_decode = []
    pad_token_id = tokenizer.pad_token_id

    for i in range(sequences.size(0)):
        start = int(prompt_lens[i].item())
        gen_tokens = sequences[i, start:]

        if pad_token_id is not None:
            valid_tokens_mask = gen_tokens != pad_token_id
            valid_len = valid_tokens_mask.sum().item()
            if valid_len > 0:
                gen_tokens = gen_tokens[:valid_len]
            else:
                gen_tokens = torch.tensor([], device=gen_tokens.device, dtype=torch.long)

        tokens_to_decode.append(gen_tokens)

    return tokenizer.batch_decode(tokens_to_decode, skip_special_tokens=True)


def _clean_prompt_texts(tokenizer, input_ids: torch.Tensor, attention_mask: torch.Tensor):
    """Return readable prompt strings by removing pads/specials via the attention mask."""
    texts = []
    for i in range(input_ids.size(0)):
        ids = input_ids[i][attention_mask[i].bool()]
        texts.append(tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False))
    return texts


# Prompts from utils.data are plain strings already; no chat templating needed here.


def _get_pass_metrics(cfg: DictConfig) -> list[PassMetric]:
    """Parse pass@ specifications supporting absolute k and proportional targets."""

    raw_values = cfg.eval.get("pass_k", []) or []
    max_generations = int(cfg.eval.get("num_generations", 0))
    metrics: list[PassMetric] = []
    seen_names: set[str] = set()

    for raw in raw_values:
        if raw is None:
            continue

        if isinstance(raw, str):
            token = raw.strip()
            if not token:
                continue
            name = token
        else:
            token = raw
            name = str(raw)

        try:
            value = float(token)
        except (TypeError, ValueError):
            continue

        if not math.isfinite(value):
            continue

        if 0 < value <= 1:
            if name in seen_names:
                continue
            metrics.append(PassMetric(name=name, is_proportion=True, value=value))
            seen_names.add(name)
            continue

        k = int(round(value))
        if k <= 0:
            continue
        if max_generations > 0 and k > max_generations:
            continue

        canonical_name = str(k)
        if canonical_name in seen_names:
            continue

        metrics.append(PassMetric(name=canonical_name, is_proportion=False, value=float(k)))
        seen_names.add(canonical_name)

    return metrics


def _pass_at_k_from_counts(n: int, c: int, k: int) -> float | None:
    """Compute pass@k for a single prompt given counts of total and correct generations."""
    if n <= 0 or k <= 0 or k > n:
        return None
    if c <= 0:
        return 0.0
    denom = comb(n, k)
    if denom == 0:
        return None
    remaining = n - c
    numer = comb(remaining, k) if remaining >= k else 0
    return 1.0 - (numer / denom)


def _update_pass_metrics(correct_flags: list[bool], pass_metrics: list[PassMetric], sums: dict[str, float], counts: dict[str, int]) -> None:
    """Accumulate pass@k-style sums/counts for averaging across prompts."""
    if not pass_metrics:
        return
    n = len(correct_flags)
    if n == 0:
        return
    c = sum(1 for flag in correct_flags if flag)
    for metric in pass_metrics:
        k = metric.k_for_count(n)
        if k <= 0 or k > n:
            continue
        value = _pass_at_k_from_counts(n, c, k)
        if value is None:
            continue
        sums[metric.name] += value
        counts[metric.name] += 1


def _evaluate_once_default(model, tokenizer, prompts, answers, cfg: DictConfig, progress_desc: str | None = None):
    """Default decoding. If cfg.eval.default_use_vllm is True, use Unsloth+vLLM
    fast_generate for better memory/throughput (like evaluate-run.py). Otherwise
    use vanilla HF generate.
    """
    # Only use Unsloth+vLLM fast_generate when both flags are enabled.
    # Prevents passing vLLM-only kwargs to HF generate when fast_inference is off.
    use_vllm = bool(cfg.eval.get("default_use_vllm", True)) and bool(cfg.model.get("fast_inference", True))
    total_queries = len(prompts)
    ans_correct = 0
    for_correct = 0
    both_correct = 0
    total_gen_len = 0
    examples = []
    pass_metrics = _get_pass_metrics(cfg)
    pass_at_metric_sums = {metric.name: 0.0 for metric in pass_metrics}
    pass_at_metric_counts = {metric.name: 0 for metric in pass_metrics}
    # Grouping and multi-sample controls unified under eval
    G = int(cfg.eval.get("batch_size_groups", 16))
    N = int(cfg.eval.get("num_generations", 1))

    if use_vllm:
        from vllm import SamplingParams
        # Map HF-style knobs to vLLM SamplingParams
        top_k = cfg.default_decode.get("top_k", 0)
        vllm_top_k = int(top_k) if int(top_k) > 0 else -1
        sampling_params = SamplingParams(
            temperature=float(cfg.default_decode.get("temperature", 1.0)),
            top_p=float(cfg.default_decode.get("top_p", 1.0)),
            top_k=vllm_top_k,
            max_tokens=int(cfg.eval.max_new_tokens),
            n=int(N),
        )
        # Track fractions over groups
        sum_ans_frac, sum_for_frac, sum_both_frac = 0.0, 0.0, 0.0
        with tqdm(total=total_queries, desc=progress_desc or "default", unit="prompt", dynamic_ncols=True) as pbar:
            for i in range(0, total_queries, G):
                j = min(i + G, total_queries)
                batch_prompts = prompts[i:j]
                outputs = model.fast_generate(
                    batch_prompts,
                    sampling_params=sampling_params,
                    lora_request=None,
                    use_tqdm=False,
                )
                for k, output, gt in zip(range(i, j), outputs, answers[i:j]):
                    # Collect N candidate outputs per prompt
                    texts_g = [oo.text for oo in output.outputs]
                    lens_g = [len(oo.token_ids) for oo in output.outputs]
                    a_list = [answer_correct(t, gt) for t in texts_g]
                    f_list = [format_correct(t) for t in texts_g]

                    _update_pass_metrics(a_list, pass_metrics, pass_at_metric_sums, pass_at_metric_counts)

                    a_any = any(a_list)
                    f_any = any(f_list)
                    both_any = any(a and f for a, f in zip(a_list, f_list))

                    ans_correct += 1 if a_any else 0
                    for_correct += 1 if f_any else 0
                    both_correct += 1 if both_any else 0

                    # Fractions per prompt
                    denom = float(len(a_list)) if len(a_list) > 0 else float(N)
                    sum_ans_frac += (sum(1 for a in a_list if a) / denom)
                    sum_for_frac += (sum(1 for f in f_list if f) / denom)
                    sum_both_frac += (sum(1 for a, f in zip(a_list, f_list) if a and f) / denom)

                    # Choose representative for logging/length
                    chosen_idx = next((idx for idx, (a, f) in enumerate(zip(a_list, f_list)) if a and f), None)
                    if chosen_idx is None:
                        chosen_idx = next((idx for idx, a in enumerate(a_list) if a), 0)
                    chosen_text = texts_g[chosen_idx]
                    chosen_len = lens_g[chosen_idx]
                    total_gen_len += chosen_len

                    if len(examples) < cfg.eval.sample_cnt:
                        examples.append({
                            "prompt": prompts[k],
                            "answer": gt,
                            "completion": chosen_text,
                            "correct": a_any,
                            "format_correct": f_any,
                        })
                pbar.update(j - i)
    else:
        # Vanilla HF generate path
        device = next(model.parameters()).device
        gen_cfg = _build_gen_config_from_section(tokenizer, cfg.default_decode, cfg.eval.max_new_tokens)
        # Track fractions over groups
        sum_ans_frac, sum_for_frac, sum_both_frac = 0.0, 0.0, 0.0
        with tqdm(total=total_queries, desc=progress_desc or "default", unit="prompt", dynamic_ncols=True) as pbar:
            for i in range(0, total_queries, G):
                j = min(i + G, total_queries)
                batch_prompts = prompts[i:j]
                enc = _batch_tokenize(tokenizer, batch_prompts, cfg.model.max_seq_length,
                                       padding_side="left", add_special_tokens=False)
                input_ids = enc.input_ids.to(device)
                attn = enc.attention_mask.to(device)
                prompt_lens = attn.sum(dim=1)

                # Repeat prompts N times to get N trajectories per prompt
                input_rep = input_ids.repeat_interleave(N, dim=0)
                attn_rep = attn.repeat_interleave(N, dim=0)

                with torch.no_grad():
                    sequences = model.generate(
                        input_ids=input_rep,
                        attention_mask=attn_rep,
                        generation_config=gen_cfg,
                    )

                # Decode all N trajectories per prompt
                prompt_lens_rep = prompt_lens.repeat_interleave(N, dim=0)
                texts_all = _decode_generated(tokenizer, sequences, prompt_lens_rep)

                # Compute generated lengths per trajectory
                if tokenizer.pad_token_id is not None:
                    seq_valid_lens_all = (sequences != tokenizer.pad_token_id).sum(dim=1)
                else:
                    seq_valid_lens_all = torch.full((sequences.size(0),), sequences.size(1), device=sequences.device)
                gen_lens_all = (seq_valid_lens_all - prompt_lens_rep).tolist()

                # Iterate per group/prompt
                for g, k in enumerate(range(i, j)):
                    gt = answers[k]
                    start = g * N
                    end = start + N
                    texts_g = texts_all[start:end]
                    gen_lens_g = gen_lens_all[start:end]

                    a_list = [answer_correct(t, gt) for t in texts_g]
                    f_list = [format_correct(t) for t in texts_g]

                    _update_pass_metrics(a_list, pass_metrics, pass_at_metric_sums, pass_at_metric_counts)

                    a_any = any(a_list)
                    f_any = any(f_list)
                    both_any = any(a and f for a, f in zip(a_list, f_list))

                    ans_correct += 1 if a_any else 0
                    for_correct += 1 if f_any else 0
                    both_correct += 1 if both_any else 0

                    sum_ans_frac += (sum(1 for a in a_list if a) / float(N))
                    sum_for_frac += (sum(1 for f in f_list if f) / float(N))
                    sum_both_frac += (sum(1 for a, f in zip(a_list, f_list) if a and f) / float(N))

                    # Representative sample for logging/lengths
                    chosen_idx = next((idx for idx, (a, f) in enumerate(zip(a_list, f_list)) if a and f), None)
                    if chosen_idx is None:
                        chosen_idx = next((idx for idx, a in enumerate(a_list) if a), 0)
                    chosen_text = texts_g[chosen_idx]
                    chosen_len = gen_lens_g[chosen_idx]
                    total_gen_len += chosen_len

                    if len(examples) < cfg.eval.sample_cnt:
                        examples.append({
                            "prompt": prompts[k],
                            "answer": gt,
                            "completion": chosen_text,
                            "correct": a_any,
                            "format_correct": f_any,
                        })
                pbar.update(j - i)

    avg_len = total_gen_len / total_queries if total_queries > 0 else 0.0
    avg_ans_frac = (sum_ans_frac / total_queries) if 'sum_ans_frac' in locals() and total_queries > 0 else 0.0
    avg_for_frac = (sum_for_frac / total_queries) if 'sum_for_frac' in locals() and total_queries > 0 else 0.0
    avg_both_frac = (sum_both_frac / total_queries) if 'sum_both_frac' in locals() and total_queries > 0 else 0.0
    avg_pass_at_k = {
        metric.name: (pass_at_metric_sums[metric.name] / pass_at_metric_counts[metric.name]) if pass_at_metric_counts[metric.name] > 0 else 0.0
        for metric in pass_metrics
    }

    return (
        ans_correct / total_queries,
        for_correct / total_queries,
        both_correct / total_queries,
        avg_len,
        avg_ans_frac,
        avg_for_frac,
        avg_both_frac,
        examples,
        avg_pass_at_k,
    )


def _is_vllm_backend(obj) -> bool:
    if obj is None:
        return False
    if hasattr(obj, "llm_engine") and hasattr(obj, "generate"):
        return True
    module_name = getattr(obj.__class__, "__module__", "")
    return module_name.startswith("vllm.") and hasattr(obj, "generate")


def _locate_vllm_backend(model) -> tuple[bool, object | None]:
    candidate_attrs = (
        "vllm_engine",
        "llm",
        "_llm",
        "llm_engine",
        "_llm_engine",
        "fast_llm",
        "fast_inference_llm",
        "vllm",
    )

    for attr in candidate_attrs:
        cand = getattr(model, attr, None)
        if _is_vllm_backend(cand):
            return True, cand

    fast_generate = getattr(model, "fast_generate", None)
    engine = getattr(fast_generate, "__self__", None) if fast_generate is not None else None
    if _is_vllm_backend(engine):
        return True, engine

    try:
        from unsloth import FastLanguageModel as _FLM  # type: ignore
        for attr in ("_fast_llm", "_llm", "llm"):
            cand = getattr(_FLM, attr, None)
            if _is_vllm_backend(cand):
                return True, cand
    except Exception:
        pass

    return False, None


def _evaluate_once_custom(model, tokenizer, prompts, answers, cfg: DictConfig, logging_enabled: bool, progress_desc: str | None = None):
    _ensure_pad_token(tokenizer)
    use_vllm_cfg = bool(cfg.model.get("fast_inference", True))
    use_vllm_detected, llm = _locate_vllm_backend(model)
    use_vllm = use_vllm_cfg and use_vllm_detected
    pass_metrics = _get_pass_metrics(cfg)

    default_decode_raw = cfg.get("default_decode", {})
    custom_decode_raw = cfg.get("custom_decode", {})
    default_decode_cfg = (
        OmegaConf.to_container(default_decode_raw, resolve=True)
        if default_decode_raw
        else {}
    )
    custom_decode_cfg = (
        OmegaConf.to_container(custom_decode_raw, resolve=True)
        if custom_decode_raw
        else {}
    )

    def _decode_param(key: str, fallback):
        if key in custom_decode_cfg and custom_decode_cfg[key] is not None:
            return custom_decode_cfg[key]
        if key in default_decode_cfg and default_decode_cfg[key] is not None:
            return default_decode_cfg[key]
        return fallback

    if use_vllm_cfg and not use_vllm_detected:
        print("[evaluate-decode] fast_inference enabled but no vLLM backend detected; falling back to HF custom generation.")

    if use_vllm:
        from trainer.vllm_smc import StepGeneration, SMCVLLM, build_prm_model

        total_queries = len(prompts)
        ans_correct = 0
        for_correct = 0
        both_correct = 0
        total_gen_len = 0
        examples: list[dict[str, object]] = []
        sum_ans_frac = 0.0
        sum_for_frac = 0.0
        sum_both_frac = 0.0
        pass_at_metric_sums = {metric.name: 0.0 for metric in pass_metrics}
        pass_at_metric_counts = {metric.name: 0 for metric in pass_metrics}
        total_group_sizes = 0
        total_groups = 0

        G = int(cfg.eval.get("batch_size_groups", 1))
        N = int(cfg.eval.get("num_generations", 1))

        confidence_cfg = custom_decode_cfg.get("confidence", {}) if custom_decode_cfg is not None else {}

        step_token = custom_decode_cfg.get("step_token")
        tokens_per_step = custom_decode_cfg.get("tokens_per_step")
        if step_token is None and tokens_per_step is None:
            step_token = "\n\n"
        if tokens_per_step is not None:
            tokens_per_step = int(tokens_per_step)
        step_token = str(step_token) if step_token is not None else None
        stop_token = custom_decode_cfg.get("stop_token", "\\boxed")
        stop_token = str(stop_token) if stop_token is not None else None
        max_steps = int(custom_decode_cfg.get("max_steps", 32))
        include_stop = bool(custom_decode_cfg.get("include_stop_str_in_output", True))
        scoring = str(confidence_cfg.get("scoring", custom_decode_cfg.get("scoring", "entropy")))
        conf_group = str(confidence_cfg.get("group", "mean"))
        conf_aggregation = str(confidence_cfg.get("aggregation", "last"))
        return_all = bool(custom_decode_cfg.get("return_all", False))
        return_eos = bool(custom_decode_cfg.get("return_eos", False))
        smc_topk = int(custom_decode_cfg.get("smc_topk", -1))
        conf_window = int(custom_decode_cfg.get("smc_confidence_window_size", 50))
        conf_eta = float(custom_decode_cfg.get("smc_confidence_eta", 1.0))
        cdf_alpha_val = confidence_cfg.get("cdf_alpha") if confidence_cfg else None
        if cdf_alpha_val is None:
            cdf_alpha_val = custom_decode_cfg.get("cdf_alpha", 0.25) if custom_decode_cfg is not None else 0.25
        cdf_alpha = float(cdf_alpha_val)
        temperature = float(_decode_param("temperature", 1.0))
        top_p = float(_decode_param("top_p", 1.0))
        top_k = int(_decode_param("top_k", 0))
        top_k = top_k if top_k > 0 else -1
        min_p = float(custom_decode_cfg.get("min_p", 0.0))
        repetition_penalty = float(custom_decode_cfg.get("repetition_penalty", 1.0))
        random_sampling = bool(custom_decode_cfg.get("random_sampling", False))
        max_new_tokens = int(cfg.eval.max_new_tokens)

        pad_token_id = tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

        # Mirror the trainer wiring (see trainer/smcgrpo_trainer.py) so decode configs stay aligned.
        step_gen = StepGeneration(
            max_steps=max_steps,
            step_token=step_token,
            tokens_per_step=tokens_per_step,
            stop_token=stop_token,
            include_stop_str_in_output=include_stop,
        )

        prm_model = None
        prm_cfg = cfg.get("prm")
        if prm_cfg and prm_cfg.get("use_prm"):
            prm_model = build_prm_model(OmegaConf.to_container(prm_cfg, resolve=True))

        smc_runner = SMCVLLM(
            llm=llm,
            tokenizer=tokenizer,
            num_particles=N,
            pad_token_id=pad_token_id,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            repetition_penalty=repetition_penalty,
            sg=step_gen,
            smc_topk=smc_topk,
            window_size=conf_window,
            confidence_eta=conf_eta,
            cdf_alpha=cdf_alpha,
            max_new_tokens=max_new_tokens,
            scoring=scoring,
            confidence_group=conf_group,
            confidence_aggregation=conf_aggregation,
            return_all=return_all,
            return_eos=return_eos,
            wandb_logging=logging_enabled,
            prm=prm_model,
            random_sampling=random_sampling,
        )

        with tqdm(total=total_queries, desc=progress_desc or "custom", unit="prompt", dynamic_ncols=True) as pbar:
            for i in range(0, total_queries, G):
                j = min(i + G, total_queries)
                batch_prompts = prompts[i:j]
                repeated_prompts: list[str] = []
                for prompt in batch_prompts:
                    repeated_prompts.extend([prompt] * N)

                outputs = smc_runner.generate(repeated_prompts, batch_prompts)
                group_sizes: List[int]
                completion_ids_list: List[List[int]]
                extras: Optional[Dict[str, Any]] = None

                if isinstance(outputs, dict):
                    extras = outputs
                elif isinstance(outputs, tuple):
                    completion_ids_list = outputs[0]
                else:
                    completion_ids_list = outputs

                if extras is not None:
                    group_completion_ids: List[List[List[int]]] = extras.get("completions", [])
                    group_sizes = extras.get(
                        "group_sizes",
                        [len(group) for group in group_completion_ids],
                    )
                    completion_ids_list = [
                        ids for group in group_completion_ids for ids in group
                    ]
                else:
                    group_sizes = [N] * (j - i)

                texts_all = tokenizer.batch_decode(completion_ids_list, skip_special_tokens=True)
                gen_lens_all = [len(ids) for ids in completion_ids_list]

                offset = 0

                for g, k in enumerate(range(i, j)):
                    gt = answers[k]
                    group_count = group_sizes[g]
                    texts_g = texts_all[offset : offset + group_count]
                    gen_lens_g = gen_lens_all[offset : offset + group_count]
                    offset += group_count

                    total_group_sizes += group_count
                    total_groups += 1

                    a_list = [answer_correct(t, gt) for t in texts_g]
                    f_list = [format_correct(t) for t in texts_g]

                    _update_pass_metrics(a_list, pass_metrics, pass_at_metric_sums, pass_at_metric_counts)

                    a_any = any(a_list)
                    f_any = any(f_list)
                    both_any = any(a and f for a, f in zip(a_list, f_list))

                    ans_correct += 1 if a_any else 0
                    for_correct += 1 if f_any else 0
                    both_correct += 1 if both_any else 0

                    denom = float(len(a_list)) if len(a_list) > 0 else float(max(group_count, 1))
                    sum_ans_frac += (sum(1 for a in a_list if a) / denom)
                    sum_for_frac += (sum(1 for f in f_list if f) / denom)
                    sum_both_frac += (sum(1 for a, f in zip(a_list, f_list) if a and f) / denom)

                    chosen_idx = next((idx for idx, (a, f) in enumerate(zip(a_list, f_list)) if a and f), None)
                    if chosen_idx is None:
                        chosen_idx = next((idx for idx, a in enumerate(a_list) if a), 0)
                    chosen_idx = 0 if chosen_idx is None else chosen_idx
                    chosen_text = texts_g[chosen_idx] if texts_g else ""
                    chosen_len = gen_lens_g[chosen_idx] if gen_lens_g else 0
                    total_gen_len += chosen_len

                    if len(examples) < cfg.eval.sample_cnt:
                        examples.append({
                            "prompt": prompts[k],
                            "answer": gt,
                            "completion": chosen_text,
                            "correct": a_any,
                            "format_correct": f_any,
                        })
                pbar.update(j - i)

        avg_group_traj = (total_group_sizes / total_groups) if total_groups > 0 else float(N)

        if return_eos and logging_enabled:
            try:
                import wandb  # type: ignore

                wandb.log({"decode/avg_group_traj": avg_group_traj})
            except ImportError:
                pass

        avg_len = total_gen_len / total_queries if total_queries > 0 else 0.0
        avg_ans_frac = sum_ans_frac / total_queries if total_queries > 0 else 0.0
        avg_for_frac = sum_for_frac / total_queries if total_queries > 0 else 0.0
        avg_both_frac = sum_both_frac / total_queries if total_queries > 0 else 0.0
        avg_pass_at_k = {
            metric.name: (pass_at_metric_sums[metric.name] / pass_at_metric_counts[metric.name]) if pass_at_metric_counts[metric.name] > 0 else 0.0
            for metric in pass_metrics
        }
        return (
            ans_correct / total_queries,
            for_correct / total_queries,
            both_correct / total_queries,
            avg_len,
            avg_ans_frac,
            avg_for_frac,
            avg_both_frac,
            examples,
            avg_pass_at_k,
        )

    device = next(model.parameters()).device

    # Build custom generation config with SMC knobs
    gen_cfg = GenerationConfig(
        do_sample=bool(_decode_param("do_sample", True)),
        temperature=float(_decode_param("temperature", 1.0)),
        top_p=float(_decode_param("top_p", 1.0)),
        top_k=int(_decode_param("top_k", 0)),
        max_new_tokens=cfg.eval.max_new_tokens,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        return_dict_in_generate=False,
    )
    # SMC flags
    gen_cfg.use_smc = bool(cfg.custom_decode.get("use_smc", True))
    # Unified eval.num_generations controls trajectories per prompt
    gen_cfg.num_generations = int(cfg.eval.get("num_generations", 1))
    # Additional SMC controls (kept for parity with training config)
    gen_cfg.smc_confidence_eta = float(custom_decode_cfg.get("smc_confidence_eta", 1.0))
    gen_cfg.smc_confidence_window_size = int(custom_decode_cfg.get("smc_confidence_window_size", 50))
    gen_cfg.smc_topk = int(custom_decode_cfg.get("smc_topk", -1))
    confidence_cfg = custom_decode_cfg.get("confidence", {}) if custom_decode_cfg is not None else {}
    gen_cfg.scoring = str(confidence_cfg.get("scoring", custom_decode_cfg.get("scoring", "entropy")))
    gen_cfg.step_token = str(custom_decode_cfg.get("step_token", "\n\n"))
    gen_cfg.stop_token = str(custom_decode_cfg.get("stop_token", "\\boxed"))
    gen_cfg.return_all = bool(custom_decode_cfg.get("return_all", False))

    # logging for custom generator
    logging_config = {
        "is_main_process": True,
        "report_to": ["wandb"] if logging_enabled else [],
        "global_step": 0,
    }

    total_queries = len(prompts)
    ans_correct = 0
    for_correct = 0
    both_correct = 0
    total_gen_len = 0
    examples = []
    # Track average fractions of correct trajectories per group
    sum_ans_frac = 0.0
    sum_for_frac = 0.0
    sum_both_frac = 0.0
    pass_at_metric_sums = {metric.name: 0.0 for metric in pass_metrics}
    pass_at_metric_counts = {metric.name: 0 for metric in pass_metrics}

    G = cfg.eval.batch_size_groups
    N = gen_cfg.num_generations

    with tqdm(total=total_queries, desc=progress_desc or "custom", unit="prompt", dynamic_ncols=True) as pbar:
        for i in range(0, total_queries, G):
            j = min(i + G, total_queries)
            # Increment global_step once per outer loop iteration (per batch of groups)
            logging_config["global_step"] = int(logging_config.get("global_step", 0)) + 1
            batch_prompts = prompts[i:j]
            enc = _batch_tokenize(tokenizer, batch_prompts, cfg.model.max_seq_length,
                                   padding_side="left", add_special_tokens=False)
            input_ids = enc.input_ids.to(device)
            attn = enc.attention_mask.to(device)
            prompt_lens = attn.sum(dim=1)

            # repeat each prompt N times for SMC groups
            input_rep = input_ids.repeat_interleave(N, dim=0)
            attn_rep = attn.repeat_interleave(N, dim=0)

            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                # Route through HF custom_generate hook so the decoding pipeline
                # (logits processors, stopping criteria, etc.) is prepared correctly.
                sequences = model.generate(
                    input_ids=input_rep,
                    attention_mask=attn_rep,
                    generation_config=gen_cfg,
                    custom_generate='.',
                    tokenizer=tokenizer,
                    # Pass clean prompt strings for debugging/inspection inside custom generator
                    prompt_texts=_clean_prompt_texts(tokenizer, input_ids, attn),
                    logging_config=logging_config,
                    trust_remote_code=True,
                )

            # Group particles per prompt: shape [groups, N, T]
            B_rep, T = sequences.shape
            assert B_rep % N == 0, "Batch size must be a multiple of num_generations"
            groups = B_rep // N
            sequences_grouped = sequences.view(groups, N, T)

            # Decode all particles for all groups
            prompt_lens_rep = prompt_lens.repeat_interleave(N, dim=0)
            texts_all = _decode_generated(tokenizer, sequences, prompt_lens_rep)

            # Compute generated lengths for all particles
            if tokenizer.pad_token_id is not None:
                seq_valid_lens_all = (sequences != tokenizer.pad_token_id).sum(dim=1)
            else:
                seq_valid_lens_all = torch.full((sequences.size(0),), sequences.size(1), device=sequences.device)
            gen_lens_all = (seq_valid_lens_all - prompt_lens_rep).tolist()

            # Iterate per-group, aggregate correctness as "any particle in group"
            for g, k in enumerate(range(i, j)):
                gt = answers[k]
                start = g * N
                end = start + N
                texts_g = texts_all[start:end]
                gen_lens_g = gen_lens_all[start:end]

                # Evaluate each particle
                a_list = [answer_correct(t, gt) for t in texts_g]
                f_list = [format_correct(t) for t in texts_g]

                _update_pass_metrics(a_list, pass_metrics, pass_at_metric_sums, pass_at_metric_counts)

                a_any = any(a_list)
                f_any = any(f_list)
                # both means there exists a particle that is both answer- and format-correct
                both_any = any(a and f for a, f in zip(a_list, f_list))

                ans_correct += 1 if a_any else 0
                for_correct += 1 if f_any else 0
                both_correct += 1 if both_any else 0

                # Accumulate fractions within this group
                sum_ans_frac += (sum(1 for a in a_list if a) / float(N))
                sum_for_frac += (sum(1 for f in f_list if f) / float(N))
                sum_both_frac += (sum(1 for a, f in zip(a_list, f_list) if a and f) / float(N))

                # Choose a representative particle for logging/lengths:
                # prefer one with both correct, else one with answer correct, else first.
                chosen_idx = next((idx for idx, (a, f) in enumerate(zip(a_list, f_list)) if a and f), None)
                if chosen_idx is None:
                    chosen_idx = next((idx for idx, a in enumerate(a_list) if a), 0)
                chosen_text = texts_g[chosen_idx]
                chosen_len = gen_lens_g[chosen_idx]
                total_gen_len += chosen_len

                if len(examples) < cfg.eval.sample_cnt:
                    examples.append({
                        "prompt": prompts[k],
                        "answer": gt,
                        "completion": chosen_text,
                        "correct": a_any,
                        "format_correct": f_any,
                    })
            pbar.update(j - i)

    avg_len = total_gen_len / total_queries if total_queries > 0 else 0.0
    avg_ans_frac = sum_ans_frac / total_queries if total_queries > 0 else 0.0
    avg_for_frac = sum_for_frac / total_queries if total_queries > 0 else 0.0
    avg_both_frac = sum_both_frac / total_queries if total_queries > 0 else 0.0
    avg_pass_at_k = {
        metric.name: (pass_at_metric_sums[metric.name] / pass_at_metric_counts[metric.name]) if pass_at_metric_counts[metric.name] > 0 else 0.0
        for metric in pass_metrics
    }
    return (
        ans_correct / total_queries,
        for_correct / total_queries,
        both_correct / total_queries,
        avg_len,
        avg_ans_frac,
        avg_for_frac,
        avg_both_frac,
        examples,
        avg_pass_at_k,
    )


def _evaluate_dataset(model, tokenizer, dataset_name: str, cfg: DictConfig, wandb_run):
    # Style selection matches training/eval convention
    style = "instruct" # if cfg.model.model_name.endswith("Instruct") else "base"
    ds = get_questions(dataset_name, split=cfg.split, style=style)
    max_prompt_len = int(getattr(cfg, "max_prompt_length", -1))
    if max_prompt_len > 0:
        def _prompt_len(text: str) -> int:
            try:
                enc = tokenizer(text, add_special_tokens=False, return_attention_mask=False)
                ids = enc.get("input_ids") or []
                return len(ids)
            except Exception:
                return len(text)
        indices: List[int] = []
        for idx, ex in enumerate(ds):
            prompt_text = ex["prompt"] if isinstance(ex, dict) else ex[0]
            if _prompt_len(prompt_text) <= max_prompt_len:
                indices.append(idx)
        if not indices:
            raise ValueError(f"No prompts with length <= {max_prompt_len} found for dataset {dataset_name}")
        if hasattr(ds, "select"):
            ds = ds.select(indices)
        else:
            ds = [ds[i] for i in indices]
    num_samples_cfg = int(getattr(cfg, "num_samples", -1))
    if num_samples_cfg > 0:
        limit = min(num_samples_cfg, len(ds))
        if hasattr(ds, "select"):
            ds = ds.select(range(limit))
        else:
            ds = ds[:limit]
    prompts = [ex["prompt"] for ex in ds]
    answers = ds["answer"]

    # Accumulate per-repeat results
    results = {}

    run_default = bool(cfg.eval.get("run_default", True))
    run_custom = bool(cfg.eval.get("run_custom", True))
    pass_metrics = _get_pass_metrics(cfg)
    pass_metric_names = [metric.name for metric in pass_metrics]

    if run_default:
        default_ans, default_for, default_both, default_len, default_ans_frac, default_for_frac, default_both_frac, default_examples = [], [], [], [], [], [], [], []
        default_pass_at_k = {name: [] for name in pass_metric_names}
    if run_custom:
        custom_ans, custom_for, custom_both, custom_len, custom_ans_frac, custom_for_frac, custom_both_frac, custom_examples = [], [], [], [], [], [], [], []
        custom_pass_at_k = {name: [] for name in pass_metric_names}

    for t in range(cfg.eval.repeat_cnt):
        if run_default:
            da, df, db, dl, dr_a, dr_f, dr_b, de, dp = _evaluate_once_default(
                model, tokenizer, prompts, answers, cfg,
                progress_desc=f"{dataset_name} | default | rep {t+1}/{cfg.eval.repeat_cnt}"
            )
            default_ans.append(da); default_for.append(df); default_both.append(db); default_len.append(dl)
            default_ans_frac.append(dr_a); default_for_frac.append(dr_f); default_both_frac.append(dr_b)
            default_examples = de if not default_examples else default_examples
            for name in pass_metric_names:
                default_pass_at_k[name].append(dp.get(name, 0.0))
        if run_custom:
            ca, cf, cb, cl, cr_a, cr_f, cr_b, ce, cp = _evaluate_once_custom(
                model, tokenizer, prompts, answers, cfg,
                logging_enabled=cfg.wandb.enable,
                progress_desc=f"{dataset_name} | custom | rep {t+1}/{cfg.eval.repeat_cnt}"
            )
            custom_ans.append(ca); custom_for.append(cf); custom_both.append(cb); custom_len.append(cl)
            custom_ans_frac.append(cr_a); custom_for_frac.append(cr_f); custom_both_frac.append(cr_b)
            custom_examples = ce if not custom_examples else custom_examples
            for name in pass_metric_names:
                custom_pass_at_k[name].append(cp.get(name, 0.0))

        # Do not log per-repeat metrics to W&B; only final means are logged later.

    if run_default:
        results["default"] = {
            "ans_acc": default_ans,
            "for_acc": default_for,
            "both_acc": default_both,
            "lengths": default_len,
            "ans_frac": default_ans_frac,
            "for_frac": default_for_frac,
            "both_frac": default_both_frac,
            "examples": default_examples,
            "pass_at_k": default_pass_at_k,
        }
    if run_custom:
        results["custom"] = {
            "ans_acc": custom_ans,
            "for_acc": custom_for,
            "both_acc": custom_both,
            "lengths": custom_len,
            "ans_frac": custom_ans_frac,
            "for_frac": custom_for_frac,
            "both_frac": custom_both_frac,
            "examples": custom_examples,
            "pass_at_k": custom_pass_at_k,
        }
    return results


@hydra.main(version_base=None, config_path="config", config_name="decode_eval")
def main(cfg: DictConfig) -> None:
    # For dataset tokenizer-dependent filtering
    set_tokenizer_name(cfg.model.model_name)

    # Prepare dated log directory and placeholder log file for this run.
    today = datetime.now().strftime("%Y-%m-%d")
    log_dir = os.path.join("outputs", today)
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "evaluate-decode.log")
    if not os.path.exists(log_path):
        with open(log_path, "w"):
            pass

    # Load model and tokenizer via Unsloth, but disable fast_inference for HF generate
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg.model.model_name,
        max_seq_length=cfg.model.max_seq_length,
        load_in_4bit=cfg.model.load_in_4bit,
        fast_inference=cfg.model.fast_inference,
        gpu_memory_utilization=cfg.model.gpu_memory_utilization,
    )

    # W&B init
    wandb_run = None
    if cfg.wandb.enable:
        import wandb
        wandb_run = wandb.init(
            project=cfg.wandb.project_name,
            name=cfg.wandb.run_name,
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    results = {}
    for dataset_name in cfg.datasets:
        print(f"Evaluating dataset: {dataset_name}")
        results[dataset_name] = _evaluate_dataset(model, tokenizer, dataset_name, cfg, wandb_run)

    # Persist results under results/decode_compare/<run_name>.json
    results_dir = os.path.join("results", "decode_compare")
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, f"{cfg.wandb.run_name}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {out_path}")

    if wandb_run is not None:
        # Log only final per-dataset mean metrics, unified across default/custom
        summary = {}
        for d, res in results.items():
            # Concatenate across available modes (default/custom) for a unified mean
            ans_vals, for_vals, both_vals, len_vals = [], [], [], []
            ans_frac_vals, for_frac_vals, both_frac_vals = [], [], []
            pass_vals: dict[str, list[float]] = {}
            for mode_data in res.values():
                ans_vals.extend(mode_data.get("ans_acc", []))
                for_vals.extend(mode_data.get("for_acc", []))
                both_vals.extend(mode_data.get("both_acc", []))
                len_vals.extend(mode_data.get("lengths", []))
                ans_frac_vals.extend(mode_data.get("ans_frac", []))
                for_frac_vals.extend(mode_data.get("for_frac", []))
                both_frac_vals.extend(mode_data.get("both_frac", []))
                for name, vals in (mode_data.get("pass_at_k", {}) or {}).items():
                    pass_vals.setdefault(name, []).extend(vals)

            def _mean(lst):
                return float(sum(lst) / len(lst)) if len(lst) > 0 else 0.0

            summary[f"{d}/ans_acc_mean"] = _mean(ans_vals)
            summary[f"{d}/for_acc_mean"] = _mean(for_vals)
            summary[f"{d}/both_acc_mean"] = _mean(both_vals)
            summary[f"{d}/length_mean"] = _mean(len_vals)
            summary[f"{d}/ans_frac_mean"] = _mean(ans_frac_vals)
            summary[f"{d}/for_frac_mean"] = _mean(for_frac_vals)
            summary[f"{d}/both_frac_mean"] = _mean(both_frac_vals)
            for name, vals in pass_vals.items():
                if vals:
                    summary[f"{d}/pass@{name}_mean"] = _mean(vals)

        wandb_run.log(summary)
        wandb_run.finish()


if __name__ == "__main__":
    main()
