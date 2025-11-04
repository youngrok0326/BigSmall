"""Run decoding for every LoRA checkpoint without reloading the base model."""

import importlib.util
import json
import os
import re
import shutil
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf

import unsloth  # Ensure Unsloth sets required flags before unsloth_zoo loads

from unsloth_zoo.vllm_utils import delete_vllm
from utils.eval import (
    ensure_shared_model,
    load_lora_request,
    resolve_base_model_path,
    unload_lora_request,
)


@dataclass
class CheckpointSpec:
    name: str
    path: str
    is_base: bool = False
def _list_lora_checkpoints(directory: str) -> List[str]:
    if not os.path.isdir(directory):
        return []
    pattern = re.compile(r"checkpoint-(\d+)")
    names = [
        name
        for name in os.listdir(directory)
        if os.path.isdir(os.path.join(directory, name)) and pattern.match(name)
    ]
    names.sort(key=lambda n: int(pattern.match(n).group(1)))
    return names


def _ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def _load_existing_results(path: str) -> Dict[str, Dict]:
    if os.path.exists(path):
        with open(path, "r") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                pass
    return {}


def _save_results(path: str, results: Dict[str, Dict]) -> None:
    _ensure_dir(os.path.dirname(path))
    with open(path, "w") as f:
        json.dump(results, f, indent=2)


def _build_wandb_run_name(prefix: str | None, checkpoint_name: str) -> str:
    if prefix:
        return f"{prefix}-{checkpoint_name}"
    return checkpoint_name


def _to_container(section) -> Dict:
    if section is None:
        return {}
    return OmegaConf.to_container(section, resolve=True)


def _build_decode_cfg(cfg: DictConfig, run_name: str) -> DictConfig:
    decode_dict: Dict[str, object] = {
        "model": _to_container(cfg.model),
        "datasets": list(cfg.datasets),
        "split": cfg.get("split"),
        "num_samples": cfg.get("num_samples"),
        "max_prompt_length": cfg.get("max_prompt_length"),
        "eval": _to_container(cfg.eval),
        "default_decode": _to_container(cfg.default_decode),
        "wandb": _to_container(cfg.wandb),
    }
    decode_dict["wandb"]["run_name"] = run_name
    return OmegaConf.create(decode_dict)


def _gather_checkpoints(cfg: DictConfig, root: str) -> List[CheckpointSpec]:
    checkpoints_root = os.path.join(root, cfg.checkpoints_dir, cfg.run_name)
    specs: List[CheckpointSpec] = []
    if cfg.include_base:
        specs.append(CheckpointSpec(name="base", path=str(cfg.base_model), is_base=True))
    for name in _list_lora_checkpoints(checkpoints_root):
        specs.append(
            CheckpointSpec(
                name=name,
                path=os.path.join(checkpoints_root, name),
                is_base=False,
            )
        )
    return specs


def _checkpoint_step(name: str) -> int:
    match = re.search(r"(\d+)", name)
    return int(match.group(1)) if match else 0


def _summarize_results(results: Dict[str, Dict[str, Dict]], prefix: str | None = None) -> Dict[str, float]:
    summary: Dict[str, float] = {}

    for dataset_name, res in results.items():
        ans_vals: List[float] = []
        for_vals: List[float] = []
        both_vals: List[float] = []
        len_vals: List[float] = []
        ans_frac_vals: List[float] = []
        for_frac_vals: List[float] = []
        both_frac_vals: List[float] = []
        pass_vals: Dict[str, List[float]] = {}

        for mode_data in res.values():
            ans_vals.extend(mode_data.get("ans_acc", []))
            for_vals.extend(mode_data.get("for_acc", []))
            both_vals.extend(mode_data.get("both_acc", []))
            len_vals.extend(mode_data.get("lengths", []))
            ans_frac_vals.extend(mode_data.get("ans_frac", []))
            for_frac_vals.extend(mode_data.get("for_frac", []))
            both_frac_vals.extend(mode_data.get("both_frac", []))
            for name, values in (mode_data.get("pass_at_k", {}) or {}).items():
                pass_vals.setdefault(name, []).extend(values)

        base_key = f"{dataset_name}/"
        if prefix:
            base_key = f"{prefix}/{base_key}"

        def _mean(seq: List[float]) -> float:
            return float(sum(seq) / len(seq)) if seq else 0.0

        summary[f"{base_key}ans_acc_mean"] = _mean(ans_vals)
        summary[f"{base_key}for_acc_mean"] = _mean(for_vals)
        summary[f"{base_key}both_acc_mean"] = _mean(both_vals)
        summary[f"{base_key}length_mean"] = _mean(len_vals)
        summary[f"{base_key}ans_frac_mean"] = _mean(ans_frac_vals)
        summary[f"{base_key}for_frac_mean"] = _mean(for_frac_vals)
        summary[f"{base_key}both_frac_mean"] = _mean(both_frac_vals)

        for name, values in pass_vals.items():
            if values:
                summary[f"{base_key}pass@{name}_mean"] = _mean(values)

    return summary


def _load_run_decode(root: str):
    module_path = os.path.join(root, "evaluate-decode.py")
    if not os.path.exists(module_path):
        raise FileNotFoundError(f"Cannot find evaluate-decode.py at {module_path}")
    spec = importlib.util.spec_from_file_location("evaluate_decode_module", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    run_decode = getattr(module, "run_decode", None)
    if run_decode is None:
        raise AttributeError("evaluate-decode.py does not define run_decode")
    return run_decode


@hydra.main(version_base=None, config_path="config", config_name="decode_run_eval")
def main(cfg: DictConfig) -> None:
    root = get_original_cwd()
    os.chdir(root)

    run_decode = _load_run_decode(root)

    checkpoints = _gather_checkpoints(cfg, root)
    if not checkpoints:
        raise RuntimeError("No checkpoints found to evaluate.")

    aggregated_dir = os.path.join(root, cfg.results_root)
    aggregated_path = os.path.join(aggregated_dir, f"{cfg.run_name}.json")
    aggregated_results: Dict[str, Dict] = _load_existing_results(aggregated_path)

    per_ckpt_dir = os.path.join(aggregated_dir, cfg.run_name)
    if cfg.save_per_checkpoint:
        _ensure_dir(per_ckpt_dir)

    decode_compare_dir = os.path.join(root, "results", "decode_compare")
    _ensure_dir(decode_compare_dir)

    wandb_run = None
    if cfg.wandb.enable:
        import wandb

        wandb_kwargs = {
            "project": cfg.wandb.project_name,
            "name": cfg.wandb.run_name,
            "config": OmegaConf.to_container(cfg, resolve=True),
        }
        if cfg.wandb.get("entity") is not None:
            wandb_kwargs["entity"] = cfg.wandb.entity
        if cfg.wandb.get("group") is not None:
            wandb_kwargs["group"] = cfg.wandb.group

        wandb_run = wandb.init(**wandb_kwargs)

    model = None
    tokenizer = None

    try:
        for spec in checkpoints:
            if cfg.skip_existing and spec.name in aggregated_results:
                print(f"Skipping {spec.name}; results already exist.")
                continue

            base_override = resolve_base_model_path(cfg.model.model_name, None if spec.is_base else spec.path)
            model, tokenizer = ensure_shared_model(base_override, cfg.model)

            run_name = _build_wandb_run_name(cfg.wandb.run_name, spec.name)
            decode_cfg = _build_decode_cfg(cfg, run_name)
            # Prevent nested W&B runs; outer script handles logging.
            if "wandb" in decode_cfg and decode_cfg.wandb is not None:
                decode_cfg.wandb.enable = False

            if "model" in decode_cfg and decode_cfg.model is not None:
                decode_cfg.model.model_name = base_override
                if hasattr(decode_cfg.model, "tokenizer_name"):
                    decode_cfg.model.tokenizer_name = base_override

            lora_request = None
            try:
                if not spec.is_base:
                    print(f"Loading LoRA adapter from {spec.path}")
                    lora_request = load_lora_request(spec.path)

                results = run_decode(
                    decode_cfg,
                    model=model,
                    tokenizer=tokenizer,
                    lora_request=lora_request,
                    wandb_run=None,
                )

                aggregated_results[spec.name] = results
                _save_results(aggregated_path, aggregated_results)
                print(f"Stored aggregated metrics for {spec.name} -> {aggregated_path}")

                if wandb_run is not None:
                    metrics = _summarize_results(results, prefix=None)
                    step = _checkpoint_step(spec.name)
                    wandb_run.log(metrics, step=step)

                run_results_path = os.path.join(decode_compare_dir, f"{run_name}.json")
                with open(run_results_path, "w") as f:
                    json.dump(results, f, indent=2)

                if cfg.save_per_checkpoint:
                    destination = os.path.join(per_ckpt_dir, f"{spec.name}.json")
                    shutil.copyfile(run_results_path, destination)
            finally:
                if lora_request is not None:
                    unload_lora_request(lora_request)

    finally:
        try:
            if model is not None:
                delete_vllm(model)
        except Exception:
            pass
        if wandb_run is not None:
            try:
                wandb_run.finish()
            except Exception:
                pass


if __name__ == "__main__":
    main()
