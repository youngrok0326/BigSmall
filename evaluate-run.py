"""
To evaluate the performance of a run by testing all LoRA checkpoints.

Developed by: Yixuan Even Xu in 2025
"""

import json
import os
import re

import hydra
from omegaconf import DictConfig

from utils.eval import test_model


def list_subdirectories(directory: str) -> list[str]:
    subdirs = [
        name
        for name in os.listdir(directory)
        if os.path.isdir(os.path.join(directory, name)) and name.startswith("checkpoint-")
    ]

    def extract_number(name: str) -> float:
        match = re.search(r"checkpoint-(\d+)", name)
        return int(match.group(1)) if match else float("inf")

    subdirs.sort(key=extract_number)
    return subdirs


@hydra.main(version_base=None, config_path="config", config_name="test")
def main(cfg: DictConfig) -> None:
    run_dir = os.path.join("checkpoints", cfg.run_name)
    lora_names = list_subdirectories(run_dir)
    os.makedirs("results", exist_ok=True)

    try:
        with open(f"results/{cfg.run_name}.json", "r") as f:
            results = json.load(f)
    except FileNotFoundError:
        results = {}

    merged_root = f"checkpoints/Merged-{cfg.run_name}"
    for lora_name in lora_names:
        path_name = os.path.join(run_dir, lora_name)
        prior = results.get(lora_name)
        results[lora_name] = test_model(cfg, path_name, merged_root, prior)
        with open(f"results/{cfg.run_name}.json", "w") as f:
            json.dump(results, f, indent=4)


if __name__ == "__main__":
    main()
