import importlib.util
import shutil
import os

def apply_patch():
    """
    Patch TRL's GRPO trainer/config with the local GRPO implementations.
    """
    try:
        spec = importlib.util.find_spec("trl")
        if spec is None or spec.origin is None:
            raise ImportError("Could not find the 'trl' library installation path.")

        trl_dir = os.path.dirname(spec.origin)
        dest_trainer_path = os.path.join(trl_dir, "trainer", "grpo_trainer.py")
        dest_config_path = os.path.join(trl_dir, "trainer", "grpo_config.py")

        source_trainer_path = os.path.join("trainer", "grpo_trainer.py")
        source_config_path = os.path.join("trainer", "grpo_config.py")

        if not os.path.exists(source_trainer_path):
            raise FileNotFoundError(f"Source trainer file not found: {source_trainer_path}")
        if not os.path.exists(source_config_path):
            raise FileNotFoundError(f"Source config file not found: {source_config_path}")

        print("Patching TRL with local GRPO implementation...")
        print(f"  - Copying {source_trainer_path} to {dest_trainer_path}")
        shutil.copyfile(source_trainer_path, dest_trainer_path)
        print(f"  - Copying {source_config_path} to {dest_config_path}")
        shutil.copyfile(source_config_path, dest_config_path)
        print("TRL patching complete.")

    except (ImportError, FileNotFoundError, Exception) as e:
        print(f"Error during TRL patching: {e}")
        raise
