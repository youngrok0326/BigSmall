"""
Utilities for toggling between vanilla vLLM scripts and the GeoGRPO-patched
variants. All patch files live under ``utils/geo_vllm/files`` and are copied
into the installed vLLM package on demand.
"""

from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path
from typing import Iterable

PATCH_ROOT = Path(__file__).resolve().parent / "files"
_TARGETS: tuple[str, ...] = (
    "vllm/lora/layers/base_linear.py",
    "vllm/lora/layers/column_parallel_linear.py",
    "vllm/lora/layers/row_parallel_linear.py",
    "vllm/lora/layers/replicated_linear.py",
    "vllm/sampling_params.py",
    "vllm/model_executor/models/qwen2.py",
    "vllm/v1/sample/metadata.py",
    "vllm/v1/worker/gpu_input_batch.py",
    "vllm/v1/outputs.py",
    "vllm/v1/worker/gpu_model_runner.py",
)
_BACKUP_SUFFIX = ".grpo_bak"


def _find_vllm_root() -> Path:
    spec = importlib.util.find_spec("vllm")
    if spec is None or spec.submodule_search_locations is None:
        raise ImportError("Could not locate the installed vllm package.")
    return Path(spec.submodule_search_locations[0])


def _candidate_site_packages() -> tuple[Path, ...]:
    vllm_dir = _find_vllm_root()
    site_root = vllm_dir.parent
    roots = [site_root]
    python_dir = site_root.parent
    lib_dir = python_dir.parent
    env_root = lib_dir.parent
    lib64_site = env_root / "lib64" / python_dir.name / site_root.name
    if lib64_site.exists() and lib64_site != site_root:
        roots.append(lib64_site)
    return tuple(dict.fromkeys(roots))


def _copy_files(pairs: Iterable[tuple[Path, Path]]) -> None:
    for src, dst in pairs:
        if not src.exists():
            raise FileNotFoundError(f"Missing patch file: {src}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def apply_geo_patch() -> None:
    """Copy GeoGRPO-patched files into the installed vLLM package."""
    for site_root in _candidate_site_packages():
        copies = []
        for rel_path in _TARGETS:
            src = PATCH_ROOT / rel_path
            dst = site_root / rel_path
            backup = dst.with_suffix(dst.suffix + _BACKUP_SUFFIX)
            if not backup.exists() and dst.exists():
                backup.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(dst, backup)
            copies.append((src, dst))
        _copy_files(copies)


def restore_vllm() -> None:
    """Restore the original vLLM files from backups if they exist."""
    for site_root in _candidate_site_packages():
        restore_pairs = []
        for rel_path in _TARGETS:
            dst = site_root / rel_path
            backup = dst.with_suffix(dst.suffix + _BACKUP_SUFFIX)
            if backup.exists():
                restore_pairs.append((backup, dst))
        if restore_pairs:
            _copy_files(restore_pairs)
