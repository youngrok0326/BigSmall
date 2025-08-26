import importlib.util
import shutil
import os

def apply_patch(algorithm: str):
    """
    Dynamically patches the TRL library's grpo_trainer.py and grpo_config.py
    with local versions based on the specified algorithm.
    """
    if not algorithm:
        print("No algorithm specified for patching. Skipping TRL patch.")
        return

    try:
        # Find the path to the installed trl library
        spec = importlib.util.find_spec("trl")
        if spec is None or spec.origin is None:
            raise ImportError("Could not find the 'trl' library installation path.")
        
        trl_dir = os.path.dirname(spec.origin)
        
        # Define destination paths in the trl library
        # We are overwriting the base GRPO files as requested.
        dest_trainer_path = os.path.join(trl_dir, "trainer", "grpo_trainer.py")
        dest_config_path = os.path.join(trl_dir, "trainer", "grpo_config.py")

        # Define source paths in the local project
        algorithm_lower = algorithm.lower()
        source_trainer_path = os.path.join("trainer", f"{algorithm_lower}_trainer.py")
        source_config_path = os.path.join("trainer", f"{algorithm_lower}_config.py")

        # Check if source files exist
        if not os.path.exists(source_trainer_path):
            raise FileNotFoundError(f"Source trainer file not found: {source_trainer_path}")
        if not os.path.exists(source_config_path):
            raise FileNotFoundError(f"Source config file not found: {source_config_path}")

        print(f"Patching TRL with '{algorithm}' implementation...")
        print(f"  - Copying {source_trainer_path} to {dest_trainer_path}")
        shutil.copyfile(source_trainer_path, dest_trainer_path)
        
        print(f"  - Copying {source_config_path} to {dest_config_path}")
        shutil.copyfile(source_config_path, dest_config_path)
        print("TRL patching complete.")

    except (ImportError, FileNotFoundError, Exception) as e:
        print(f"Error during TRL patching: {e}")
        raise