"""Helper utilities shared across SMC training components."""

from __future__ import annotations

import inspect
import os
import types
from typing import Any, Dict, List, Mapping, Optional, Tuple

from omegaconf import DictConfig, OmegaConf

def encode_batch(tokenizer: Any, texts: List[str]) -> List[List[int]]:
    tok = getattr(tokenizer, "tokenizer", None) or tokenizer
    enc = tok(texts, add_special_tokens=False, padding=False, return_attention_mask=False)  # type: ignore[misc]
    ids = enc["input_ids"] if isinstance(enc, dict) else getattr(enc, "input_ids", enc)
    return ids

def to_plain_dict(data: Optional[Mapping[str, Any] | Any]) -> Dict[str, Any]:
    if data is None:
        return {}
    if isinstance(data, dict):
        return dict(data)
    if isinstance(data, DictConfig):  # type: ignore[arg-type]
        return OmegaConf.to_container(data, resolve=True)  # type: ignore[no-any-return]
    if hasattr(data, "items"):
        return dict(data.items())
    return dict(data)

def parse_cuda_visible_from_device(device: str | int | None) -> Optional[str]:
    if device is None:
        return None
    if isinstance(device, int):
        return str(device)
    s = str(device).strip()

    if s.isdigit():
        return s
    if s.lower().startswith("cuda:"):
        return s.split(":", 1)[1]
    return None

def resolve_prm_device(cfg: Mapping[str, Any]) -> Tuple[str, Optional[str]]:
    visible_env = os.environ.get("CUDA_VISIBLE_DEVICES")
    visible_list = [v.strip() for v in visible_env.split(",") if v.strip()] if visible_env else []
    visible_set = set(visible_list)
    target_physical: Optional[str] = None

    if cfg.get("cuda") is not None:
        target_physical = str(int(cfg.get("cuda")))
    else:
        dev_cfg = cfg.get("device")

        if isinstance(dev_cfg, int):
            target_physical = str(dev_cfg)
        elif isinstance(dev_cfg, str):
            dev = dev_cfg.strip()
            lower_dev = dev.lower()

            if lower_dev.startswith("cuda:"):
                suffix = dev.split(":", 1)[1].strip()

                if suffix.isdigit():
                    target_physical = suffix
                else:
                    return dev, None
            elif dev.isdigit():
                target_physical = dev
            else:
                return dev, None
    if target_physical is None:
        return "cuda:0", None
    if not visible_list:
        return f"cuda:{target_physical}", None
    if target_physical in visible_set:
        local_idx = visible_list.index(target_physical)
        return f"cuda:{local_idx}", None
    return "cuda:0", target_physical

_PRM_ENV_FLAGS = {
    "VLLM_USE_V1": "1",
    "VLLM_TRY_V1": "1",
    "VLLM_ALLOW_LONG_MAX_MODEL_LEN": "1",
}

def apply_prm_env_flags() -> None:
    os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
    for key, value in _PRM_ENV_FLAGS.items():
        os.environ[key] = value


def patch_reward_module(reward_module: Any) -> None:
    if not getattr(reward_module, "_smcgrpo_llm_device_patch", False):
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

                reward_module.LLM = _llm_without_device  # type: ignore[assignment]
        reward_module._smcgrpo_llm_device_patch = True

    if getattr(reward_module, "_smcgrpo_reward_pooling_patch", False):
        return

    VllmProcessRewardModel = reward_module.VllmProcessRewardModel
    original_init = VllmProcessRewardModel.__init__

    def _patched_init(self, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        model = getattr(self, "model", None)
        encode_fn = getattr(model, "encode", None)

        if model is None or not callable(encode_fn):
            return
        if getattr(encode_fn, "_smcgrpo_reward_pooling_patch", False):
            return

        original_encode = getattr(encode_fn, "__func__", encode_fn)
        state = {"force_encode": False}

        def _encode_with_reward(self: Any, *enc_args: Any, **enc_kwargs: Any) -> Any:
            if state["force_encode"]:
                kwargs = dict(enc_kwargs)
                kwargs["pooling_task"] = "encode"
                return original_encode(self, *enc_args, **kwargs)

            kwargs = dict(enc_kwargs)
            kwargs["pooling_task"] = "reward"

            try:
                return original_encode(self, *enc_args, **kwargs)
            except (TypeError, ValueError) as exc:
                if "pooling_task" not in str(exc):
                    raise
                state["force_encode"] = True
                kwargs["pooling_task"] = "encode"
                return original_encode(self, *enc_args, **kwargs)

        _encode_with_reward._smcgrpo_reward_pooling_patch = True  # type: ignore[attr-defined]
        model.encode = types.MethodType(_encode_with_reward, model)

    VllmProcessRewardModel.__init__ = _patched_init  # type: ignore[assignment]
    reward_module._smcgrpo_reward_pooling_patch = True
