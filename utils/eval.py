"""
Implementation of the evaluation helpers for testing language models on math datasets.

Developed by: Yixuan Even Xu in 2025
"""

import atexit
import contextlib
import gc
import json
import os
from types import MethodType
from typing import Dict, Optional, Tuple

import torch
from omegaconf import DictConfig
from peft import PeftConfig
from unsloth import FastLanguageModel

from unsloth_zoo.vllm_utils import (
    delete_vllm,
    load_lora as unsloth_load_lora,
    prepare_vllm_lora_loading,
)


def _build_stop_token_ids(tokenizer) -> list[int]:
    """Collect EOS-like token ids so vLLM can stop early when possible."""

    stop_token_ids: list[int] = []
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if isinstance(eos_id, int) and eos_id >= 0:
        stop_token_ids.append(eos_id)

    for token in ("<|im_end|>", "</s>"):
        try:
            token_id = tokenizer.convert_tokens_to_ids(token)
        except Exception:  # pragma: no cover - defensive
            token_id = None
        if isinstance(token_id, int) and token_id >= 0 and token_id not in stop_token_ids:
            stop_token_ids.append(token_id)
    return stop_token_ids


def _resolve_base_model_path(default_base_model: str, lora_path: Optional[str]) -> str:
    if not lora_path:
        return default_base_model

    adapter_config_path = os.path.join(lora_path, "adapter_config.json")
    if os.path.isfile(adapter_config_path):
        try:
            with open(adapter_config_path, "r") as f:
                adapter_cfg = json.load(f)
            base_model = adapter_cfg.get("base_model_name_or_path")
            if isinstance(base_model, str) and base_model.strip():
                return base_model
        except Exception as exc:  # pragma: no cover - defensive
            print(f"Warning: failed to read {adapter_config_path}: {exc}")
    return default_base_model


def _model_loading_kwargs(cfg: DictConfig) -> Dict[str, object]:
    """Extract common FastLanguageModel kwargs from the config."""

    max_seq_length = int(getattr(cfg, "max_seq_length", 1024))
    load_in_4bit = bool(getattr(cfg, "load_in_4bit", True))
    fast_inference = bool(getattr(cfg, "fast_inference", True))
    gpu_memory_utilization = float(getattr(cfg, "gpu_memory_utilization", 0.5))
    return {
        "max_seq_length": max_seq_length,
        "load_in_4bit": load_in_4bit,
        "fast_inference": fast_inference,
        "gpu_memory_utilization": gpu_memory_utilization,
    }


class _SharedModelHost:
    """Keep a single FastLanguageModel instance alive across checkpoint sweeps."""

    def __init__(self) -> None:
        self.model = None
        self.tokenizer = None
        self.base_model = None
        self._lora_shell_ready = False
        self._lora_signature: Optional[Tuple[int, int, Tuple[str, ...]]] = None

    def ensure_model(self, base_model: str, cfg: DictConfig) -> Tuple[object, object]:
        if self.model is not None and self.base_model == base_model:
            return self.model, self.tokenizer

        self.release()
        kwargs = _model_loading_kwargs(cfg)
        model, tokenizer = FastLanguageModel.from_pretrained(base_model, **kwargs)
        self.model = model
        self.tokenizer = tokenizer
        self.base_model = base_model
        self._lora_shell_ready = False
        self._lora_signature = None
        return model, tokenizer

    def _prepare_lora_shell(self, adapter_path: str) -> PeftConfig:
        if self.model is None:
            raise RuntimeError("Base model must be loaded before preparing LoRA shell.")

        peft_cfg = PeftConfig.from_pretrained(adapter_path)
        signature = (
            int(peft_cfg.r),
            int(peft_cfg.lora_alpha),
            tuple(sorted(peft_cfg.target_modules)),
        )
        if not self._lora_shell_ready or signature != self._lora_signature:
            model = FastLanguageModel.get_peft_model(
                self.model,
                r=peft_cfg.r,
                lora_alpha=peft_cfg.lora_alpha,
                target_modules=list(peft_cfg.target_modules),
                lora_dropout=getattr(peft_cfg, "lora_dropout", 0.0),
                use_gradient_checkpointing="unsloth",
                random_state=getattr(peft_cfg, "random_state", 3407),
            )
            prepare_vllm_lora_loading(model)
            self.model = model
            self._lora_shell_ready = True
            self._lora_signature = signature
        return peft_cfg

    def _ensure_load_lora(self) -> object:
        if self.model is None:
            raise RuntimeError("Base model must be loaded before applying LoRA.")
        if hasattr(self.model, "load_lora"):
            return self.model

        def _load_lora(self, adapter_path: str, *, load_tensors: bool = False):
            return unsloth_load_lora(self, adapter_path, load_tensors=load_tensors)

        self.model.load_lora = MethodType(_load_lora, self.model)
        return self.model

    def load_lora(self, adapter_path: str):
        self._prepare_lora_shell(adapter_path)
        model = self._ensure_load_lora()
        return model.load_lora(adapter_path)

    def unload_lora(self, lora_request) -> None:
        if self.model is None or lora_request is None:
            return

        engine = getattr(self.model, "vllm_engine", None)
        candidates = [engine]
        if engine is not None:
            candidates.append(getattr(engine, "llm_engine", None))

        for candidate in candidates:
            if candidate is None:
                continue
            remove = getattr(candidate, "remove_lora", None)
            if remove is not None:
                with contextlib.suppress(Exception):
                    remove(lora_request.lora_int_id)
            reset = getattr(candidate, "reset_prefix_cache", None)
            if reset is not None:
                with contextlib.suppress(Exception):
                    reset()

    def release(self) -> None:
        if self.model is None:
            return
        with contextlib.suppress(Exception):
            delete_vllm(self.model)
        with contextlib.suppress(Exception):
            del self.model
        with contextlib.suppress(Exception):
            del self.tokenizer
        self.model = None
        self.tokenizer = None
        self.base_model = None
        self._lora_shell_ready = False
        self._lora_signature = None
        gc.collect()
        with contextlib.suppress(Exception):
            torch.cuda.empty_cache()


_MODEL_HOST = _SharedModelHost()
atexit.register(_MODEL_HOST.release)


def test(
    cfg: DictConfig,
    model,
    tokenizer,
    dataset,
    results: Optional[dict],
    *,
    lora_request=None,
) -> Dict[str, object]:
    from vllm import SamplingParams
    from utils.data import answer_correct, format_correct

    sampling_params = SamplingParams(
        temperature=float(cfg.temperature),
        max_tokens=int(cfg.max_tokens),
        stop_token_ids=_build_stop_token_ids(tokenizer) or None,
    )

    finished = 0
    ans_acc = []
    for_acc = []
    both_acc = []
    lengths = []
    examples = []

    if results is not None:
        finished = len(results.get("ans_acc", []))
        ans_acc = list(results.get("ans_acc", []))
        for_acc = list(results.get("for_acc", []))
        both_acc = list(results.get("both_acc", []))
        lengths = list(results.get("lengths", []))
        examples = list(results.get("examples", []))

    prompts = [entry["prompt"] for entry in dataset]
    answers = dataset["answer"]

    for t in range(finished, int(cfg.repeat_cnt)):
        print(f"Testing repeat {t}...")
        total_queries = len(prompts)
        total_length = 0
        ans_correct = 0
        for_correct = 0
        both_correct = 0

        for i in range(0, total_queries, int(cfg.batch_size)):
            j = min(i + int(cfg.batch_size), total_queries)
            batch_prompts = prompts[i:j]
            outputs = model.fast_generate(
                batch_prompts,
                sampling_params=sampling_params,
                lora_request=lora_request,
                use_tqdm=False,
            )
            for k, output, answer in zip(range(i, j), outputs, answers[i:j]):
                text = output.outputs[0].text
                token_ids = output.outputs[0].token_ids
                total_length += len(token_ids)
                ans_ok = answer_correct(text, answer)
                fmt_ok = format_correct(text)
                ans_correct += ans_ok
                for_correct += fmt_ok
                both = ans_ok and fmt_ok
                both_correct += both
                if len(examples) < int(cfg.sample_cnt):
                    examples.append(
                        {
                            "prompt": prompts[k],
                            "answer": answer,
                            "completion": text,
                            "correct": ans_ok,
                            "format_correct": fmt_ok,
                        }
                    )
        lengths.append(total_length / total_queries if total_queries else 0.0)
        ans_acc.append(ans_correct / total_queries if total_queries else 0.0)
        for_acc.append(for_correct / total_queries if total_queries else 0.0)
        both_acc.append(both_correct / total_queries if total_queries else 0.0)

        print(f"Answer accuracy for repeat {t}: {ans_acc[-1]:.2%}")
        print(f"Format accuracy for repeat {t}: {for_acc[-1]:.2%}")
        print(f"Both accuracy for repeat {t}: {both_acc[-1]:.2%}")
        print(f"Average completion length for repeat {t}: {lengths[-1]:.2f}")

    return {
        "ans_acc": ans_acc,
        "for_acc": for_acc,
        "both_acc": both_acc,
        "lengths": lengths,
        "examples": examples,
    }


def test_model(
    cfg: DictConfig,
    lora_name: str,
    merged_directory: str,  # kept for compatibility; no longer used
    results: Optional[dict],
) -> Dict[str, dict]:
    finished = True
    if results is not None:
        for dataset_name in cfg.datasets:
            dataset_res = results.get(dataset_name)
            if dataset_res is None or len(dataset_res.get("ans_acc", [])) < int(cfg.repeat_cnt):
                finished = False
                break
    else:
        finished = False

    if finished:
        return results

    base_model_for_eval = _resolve_base_model_path(cfg.base_model, lora_name or None)
    model, tokenizer = _MODEL_HOST.ensure_model(base_model_for_eval, cfg)

    from utils.data import get_questions, set_tokenizer_name

    set_tokenizer_name(cfg.base_model)

    lora_request = None
    if lora_name:
        print(f"Testing LoRA checkpoint {lora_name}...")
        lora_request = _MODEL_HOST.load_lora(lora_name)
    else:
        print("Testing base model...")

    results = {} if results is None else dict(results)

    try:
        for dataset_name in cfg.datasets:
            print(f"Testing dataset {dataset_name}...")
            dataset_testing = get_questions(dataset_name, split="test")
            prev = results.get(dataset_name)
            results[dataset_name] = test(
                cfg,
                model,
                tokenizer,
                dataset_testing,
                prev,
                lora_request=lora_request,
            )
        return results
    finally:
        _MODEL_HOST.unload_lora(lora_request)


def ensure_shared_model(base_model: str, cfg_section: DictConfig):
    """Expose the shared model host for scripts that need manual control."""

    return _MODEL_HOST.ensure_model(base_model, cfg_section)


def load_lora_request(adapter_path: str):
    """Load a LoRA adapter onto the shared model host and return the vLLM request."""

    return _MODEL_HOST.load_lora(adapter_path)


def unload_lora_request(lora_request) -> None:
    """Detach a previously loaded LoRA adapter from the shared model host."""

    _MODEL_HOST.unload_lora(lora_request)


def resolve_base_model_path(default_base_model: str, lora_path: Optional[str]) -> str:
    """Public wrapper for resolving the source model stored in a LoRA adapter."""

    return _resolve_base_model_path(default_base_model, lora_path)
