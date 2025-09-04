"""
To evaluate the performance of a run by testing all LoRA checkpoints.

"""

import re
import os
import json
import hydra
from omegaconf import DictConfig
from utils.eval import test_model

def list_subdirectories(directory):
    if not os.path.exists(directory):
        print(f"Warning: Directory {directory} does not exist.")
        return []
        
    subdirs = [name for name in os.listdir(directory) 
               if os.path.isdir(os.path.join(directory, name)) and name.startswith("checkpoint-")]
    def extract_number(name):
        match = re.search(r"checkpoint-(\d+)", name)
        return int(match.group(1)) if match else float('inf')
    subdirs.sort(key=extract_number)
    return subdirs

@hydra.main(version_base=None, config_path="config", config_name="test")
def main(cfg: DictConfig) -> None:
    from utils.data import set_tokenizer_name
    set_tokenizer_name(cfg.base_model)
    
    # Create results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)
    
    lora_names = list_subdirectories(f"checkpoints/{cfg.run_name}")
    if not lora_names:
        print(f"No checkpoint directories found in checkpoints/{cfg.run_name}")
        return
    
    try:
        with open(f"results/{cfg.run_name}.json", "r") as f:
            results = json.load(f)
    except FileNotFoundError:
        results = {}
        
    for lora_name in lora_names:
        path_name = f"checkpoints/{cfg.run_name}/{lora_name}"
        try:
            if lora_name in results:
                results[lora_name] = test_model(cfg, path_name, f"checkpoints/Merged-{cfg.run_name}", results[lora_name])
            else:
                results[lora_name] = test_model(cfg, path_name, f"checkpoints/Merged-{cfg.run_name}", None)
        except Exception as e:
            print(f"Error processing checkpoint {lora_name}: {e}")
            results[lora_name] = {"error": str(e)}
            
        # Save results after each checkpoint to avoid losing progress
        with open(f"results/{cfg.run_name}.json", "w") as f:
            json.dump(results, f, indent=4)

if __name__ == "__main__":
    main()