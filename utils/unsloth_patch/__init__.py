"""
Utilities for patching Unsloth-specific files at runtime.
"""

from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path
from typing import Iterable
import sys

PATCH_ROOT = Path(__file__).resolve().parent / "files"
_TARGETS: tuple[str, ...] = (
    "unsloth_zoo/rl_replacements.py",
    "unsloth/models/rl_replacements.py",
)
_BACKUP_SUFFIX = ".grpo_bak"


def _candidate_site_packages() -> tuple[Path, ...]:
    spec = importlib.util.find_spec("unsloth_zoo")
    if spec is None or spec.submodule_search_locations is None:
        raise ImportError("Could not locate the installed unsloth_zoo package.")
    pkg_dir = Path(spec.submodule_search_locations[0])
    site_root = pkg_dir.parent
    roots = [site_root]
    python_dir = site_root.parent
    lib_dir = python_dir.parent
    env_root = lib_dir.parent
    lib64_site = env_root / "lib64" / python_dir.name / site_root.name
    if lib64_site.exists() and lib64_site != site_root:
        roots.append(lib64_site)
    # Also consider local .venv paths if they exist.
    cwd = Path.cwd()
    for base in ("lib", "lib64"):
        local_site = cwd / ".venv" / base / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
        if local_site.exists():
            roots.append(local_site)
    return tuple(dict.fromkeys(roots))


def _copy_files(pairs: Iterable[tuple[Path, Path]]) -> None:
    for src, dst in pairs:
        if not src.exists():
            raise FileNotFoundError(f"Missing patch file: {src}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def apply_unsloth_patch() -> None:
    """Copy patched Unsloth files into the installed package."""
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


def restore_unsloth() -> None:
    """Restore original Unsloth files from backups."""
    for site_root in _candidate_site_packages():
        restore_pairs = []
        for rel_path in _TARGETS:
            dst = site_root / rel_path
            backup = dst.with_suffix(dst.suffix + _BACKUP_SUFFIX)
            if backup.exists():
                restore_pairs.append((backup, dst))
        if restore_pairs:
            _copy_files(restore_pairs)
