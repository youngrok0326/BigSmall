"""Run custom decode evaluation across all LoRA checkpoints.

This orchestrates LoRA merging and delegates per-checkpoint decoding to
``evaluate-decode.py`` so we can reuse its custom decoding implementation
and W&B logging without reinitialising the job manually for each checkpoint.
"""

import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig


@dataclass
class CheckpointSpec:
    name: str
    path: str
    requires_merge: bool


def _list_lora_checkpoints(directory: str) -> List[str]:
    """Return checkpoint subdirectories sorted by the numeric step."""

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


def _merge_lora(base_model: str, lora_path: str, merged_dir: str, root: str) -> None:
    if os.path.isdir(merged_dir):
        shutil.rmtree(merged_dir)
    _ensure_dir(merged_dir)
    subprocess.run(
        [
            "python3",
            os.path.join(root, "scripts", "merge.py"),
            base_model,
            lora_path,
            merged_dir,
        ],
        check=True,
        cwd=root,
    )


def _build_wandb_run_name(prefix: Optional[str], checkpoint_name: str) -> str:
    if prefix:
        return f"{prefix}-{checkpoint_name}"
    return checkpoint_name


def _collect_overrides(
    cfg: DictConfig,
    model_dir: str,
    run_name: str,
) -> List[str]:
    overrides: List[str] = [
        f"model.model_name={model_dir}",
        f"wandb.run_name={run_name}",
        f"wandb.project_name={cfg.wandb.project_name}",
        f"wandb.enable={'true' if cfg.wandb.enable else 'false'}",
        "hydra.run.dir=.",
        "hydra.output_subdir=null",
        "hydra.job.chdir=false",
    ]
    if cfg.wandb.get("entity"):
        overrides.append(f"wandb.entity={cfg.wandb.entity}")
    if cfg.wandb.get("group"):
        overrides.append(f"wandb.group={cfg.wandb.group}")
    extra = cfg.get("evaluate_decode_overrides")
    if extra:
        overrides.extend(list(extra))
    return overrides


def _run_decode(
    root: str,
    overrides: List[str],
) -> Tuple[Dict, str]:
    cmd = ["python3", os.path.join(root, "evaluate-decode.py"), *overrides]
    subprocess.run(cmd, check=True, cwd=root)
    run_name = None
    for override in overrides:
        if override.startswith("wandb.run_name="):
            run_name = override.split("=", 1)[1]
            break
    if not run_name:
        raise RuntimeError("wandb.run_name override missing")
    result_path = os.path.join(root, "results", "decode_compare", f"{run_name}.json")
    if not os.path.exists(result_path):
        raise FileNotFoundError(f"Expected results at {result_path}")
    with open(result_path, "r") as f:
        return json.load(f), result_path


def _gather_checkpoints(cfg: DictConfig, root: str) -> List[CheckpointSpec]:
    checkpoints_root = os.path.join(root, cfg.checkpoints_dir, cfg.run_name)
    checkpoints: List[CheckpointSpec] = []
    if cfg.include_base:
        checkpoints.append(
            CheckpointSpec(name="base", path=cfg.base_model, requires_merge=False)
        )
    for name in _list_lora_checkpoints(checkpoints_root):
        checkpoints.append(
            CheckpointSpec(
                name=name,
                path=os.path.join(checkpoints_root, name),
                requires_merge=True,
            )
        )
    return checkpoints


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

    for spec in checkpoints:
        if cfg.skip_existing and spec.name in aggregated_results:
            print(f"Skipping {spec.name}; results already exist.")
            continue

        print(f"Evaluating checkpoint: {spec.name}")
        merged_dir: Optional[str] = None
        model_dir = spec.path

        if spec.requires_merge:
            merged_dir = os.path.join(root, cfg.merged_root, f"{spec.name}_merged")
            _merge_lora(cfg.base_model, spec.path, merged_dir, root)
            model_dir = merged_dir

        run_name = _build_wandb_run_name(cfg.wandb.run_name, spec.name)
        overrides = _collect_overrides(cfg, model_dir, run_name)

        try:
            results, source_path = _run_decode(root, overrides)
        finally:
            if merged_dir and os.path.isdir(merged_dir):
                shutil.rmtree(merged_dir)

        aggregated_results[spec.name] = results
        _save_results(aggregated_path, aggregated_results)
        print(f"Stored aggregated metrics for {spec.name} -> {aggregated_path}")

        if cfg.save_per_checkpoint:
            destination = os.path.join(per_ckpt_dir, f"{spec.name}.json")
            shutil.copyfile(source_path, destination)


if __name__ == "__main__":
    main()
