"""Run custom decoding for every LoRA checkpoint without reloading the base model."""

import json
import os
import re
import shutil
from dataclasses import dataclass
from typing import Dict, List

import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf

from evaluate_decode import run_decode
from unsloth import FastLanguageModel
from unsloth_zoo.vllm_utils import load_lora, delete_vllm


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
        "custom_decode": _to_container(cfg.custom_decode),
        "prm": _to_container(cfg.prm),
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


@hydra.main(version_base=None, config_path="config", config_name="decode_run_eval")
def main(cfg: DictConfig) -> None:
    root = get_original_cwd()
    os.chdir(root)

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

    model_cfg = cfg.model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_cfg.model_name,
        max_seq_length=model_cfg.max_seq_length,
        load_in_4bit=model_cfg.load_in_4bit,
        fast_inference=model_cfg.fast_inference,
        gpu_memory_utilization=model_cfg.gpu_memory_utilization,
    )

    try:
        for spec in checkpoints:
            if cfg.skip_existing and spec.name in aggregated_results:
                print(f"Skipping {spec.name}; results already exist.")
                continue

            run_name = _build_wandb_run_name(cfg.wandb.run_name, spec.name)
            decode_cfg = _build_decode_cfg(cfg, run_name)

            lora_request = None
            if not spec.is_base:
                print(f"Loading LoRA adapter from {spec.path}")
                lora_request = load_lora(model, spec.path, load_tensors=False)

            results = run_decode(
                decode_cfg,
                model=model,
                tokenizer=tokenizer,
                lora_request=lora_request,
            )

            aggregated_results[spec.name] = results
            _save_results(aggregated_path, aggregated_results)
            print(f"Stored aggregated metrics for {spec.name} -> {aggregated_path}")

            run_results_path = os.path.join(decode_compare_dir, f"{run_name}.json")
            with open(run_results_path, "w") as f:
                json.dump(results, f, indent=2)

            if cfg.save_per_checkpoint:
                destination = os.path.join(per_ckpt_dir, f"{spec.name}.json")
                shutil.copyfile(run_results_path, destination)

    finally:
        try:
            delete_vllm(model)
        except Exception:
            pass


if __name__ == "__main__":
    main()
