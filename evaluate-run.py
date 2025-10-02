"""
To evaluate the performance of a run by testing all LoRA checkpoints.

Developed by: Yixuan Even Xu in 2025
"""

import json
import os
import re
from typing import Optional, Tuple

import hydra
from omegaconf import DictConfig
from peft import PeftConfig
from peft.utils.save_and_load import load_peft_weights, set_peft_model_state_dict

from utils.eval import test

def list_subdirectories(directory):
    subdirs = [name for name in os.listdir(directory) 
               if os.path.isdir(os.path.join(directory, name)) and name.startswith("checkpoint-")]
    def extract_number(name):
        match = re.search(r"checkpoint-(\d+)", name)
        return int(match.group(1)) if match else float('inf')
    subdirs.sort(key=extract_number)
    return subdirs


def _prepare_lora_host(model, adapter_path: str) -> Tuple[object, PeftConfig]:
    """Attach a PEFT adapter shell and cache vLLM <-> HF LoRA tensors."""

    peft_cfg = PeftConfig.from_pretrained(adapter_path)

    if not getattr(model, "_lora_shell_initialized", False):
        from unsloth import FastLanguageModel  # local import to avoid circulars at module load

        model = FastLanguageModel.get_peft_model(
            model,
            r=peft_cfg.r,
            lora_alpha=peft_cfg.lora_alpha,
            target_modules=list(peft_cfg.target_modules),
            lora_dropout=getattr(peft_cfg, "lora_dropout", 0.0),
            use_gradient_checkpointing="unsloth",
            random_state=getattr(peft_cfg, "random_state", 3407),
        )

        from unsloth_zoo.vllm_utils import prepare_vllm_lora_loading

        prepare_vllm_lora_loading(model)
        model._lora_shell_initialized = True

    return model, peft_cfg


def _load_lora_into_vllm(model, adapter_path: str, device: Optional[str] = None) -> None:
    """Load LoRA weights from disk into the model and mirror them to vLLM."""

    weights = load_peft_weights(adapter_path, device=device)
    set_peft_model_state_dict(model, weights, adapter_name="default")
    del weights
    model.set_adapter("default")

    from unsloth_zoo.vllm_utils import load_lora_directly

    load_lora_directly(model)

    if hasattr(model, "vllm_engine"):
        model.vllm_engine.reset_prefix_cache()

@hydra.main(version_base=None, config_path="config", config_name="test")
def main(cfg: DictConfig) -> None:
    from utils.data import set_tokenizer_name
    set_tokenizer_name(cfg.base_model)
    lora_names = list_subdirectories(f"checkpoints/{cfg.run_name}")
    # Ensure output directory exists to avoid FileNotFoundError
    os.makedirs("results", exist_ok=True)
    try:
        with open(f"results/{cfg.run_name}.json", "r") as f:
            results = json.load(f)
    except FileNotFoundError:
        results = {}

    # Load a single vLLM-backed model once (no per-checkpoint engine reinit)
    from utils.data import get_questions
    from unsloth_zoo.vllm_utils import delete_vllm
    from unsloth import FastLanguageModel

    model, tokenizer = None, None
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            cfg.base_model,
            max_seq_length=cfg.max_seq_length,
            load_in_4bit=True,
            fast_inference=True,
            gpu_memory_utilization=0.5,
        )

        for lora_name in lora_names:
            path_name = f"checkpoints/{cfg.run_name}/{lora_name}"
            print(f"Evaluating with LoRA adapter: {lora_name}")

            model, _ = _prepare_lora_host(model, path_name)
            _load_lora_into_vllm(model, path_name)

            ckpt_res = results.get(lora_name, {})
            for dataset_name in cfg.datasets:
                print(f"Testing dataset {dataset_name}...")
                dataset_testing = get_questions(dataset_name, split="test")

                prev = ckpt_res.get(dataset_name)
                ckpt_res[dataset_name] = test(cfg, model, dataset_testing, prev, lora_request=None)

            results[lora_name] = ckpt_res
            with open(f"results/{cfg.run_name}.json", "w") as f:
                json.dump(results, f, indent=4)
    finally:
        # Ensure the vLLM engine is destroyed only once at the end
        try:
            delete_vllm(model)
        except Exception:
            pass

if __name__ == "__main__":
    main()
