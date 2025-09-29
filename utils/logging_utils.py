"""Helpers to keep run-time log files inside the configured outputs directory."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional


def _resolve_output_dir() -> Path:
    """Return the absolute run directory Hydra configured, or fall back to ./outputs."""

    # Try to obtain the active Hydra run directory if Hydra is managing this process.
    try:  # pragma: no cover - hydra may not be present during some unit tests
        from hydra.core.hydra_config import HydraConfig  # type: ignore

        hydra_config = HydraConfig.get()
        if hydra_config is not None:
            runtime_dir = getattr(hydra_config.runtime, "output_dir", None)
            if runtime_dir:
                return Path(runtime_dir).resolve()
    except Exception:
        # Hydra is either not initialised yet or unavailable; fall back below.
        pass

    env_root = os.environ.get("HYDRA_ORIGINAL_CWD")
    if env_root:
        return Path(env_root).joinpath("outputs").resolve()

    # Default to a deterministic `outputs` folder relative to the current cwd.
    return Path(os.getcwd()).joinpath("outputs").resolve()


def ensure_log_file(filename: str, *, subdir: Optional[str] = None) -> Path:
    """Guarantee that `filename` exists inside the run's outputs directory.

    Parameters
    ----------
    filename:
        Name of the log file (e.g. ``"train.log"``).
    subdir:
        Optional subdirectory inside the outputs directory. This is primarily
        available for future customisation; current callers keep everything at
        the root of the run directory.
    """

    base_dir = _resolve_output_dir()
    if subdir:
        base_dir = base_dir.joinpath(subdir)

    base_dir.mkdir(parents=True, exist_ok=True)
    log_path = base_dir.joinpath(filename)
    log_path.touch(exist_ok=True)
    return log_path


def setup_file_logging(filename: str, *, level: int = logging.INFO) -> Path:
    """Attach a file handler for the root logger that writes to the outputs dir."""

    log_path = ensure_log_file(filename)

    root_logger = logging.getLogger()
    absolute_target = str(log_path)
    already_configured = any(
        isinstance(handler, logging.FileHandler)
        and getattr(handler, "baseFilename", None) == absolute_target
        for handler in root_logger.handlers
    )

    if not already_configured:
        file_handler = logging.FileHandler(absolute_target)
        file_handler.setLevel(level)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    return log_path
