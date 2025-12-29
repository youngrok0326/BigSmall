import importlib.util
import shutil
import os

def apply_patch():
    """
    Patch TRL's GRPO/IVO trainer/config with the local implementations.
    """
    try:
        spec = importlib.util.find_spec("trl")
        if spec is None or spec.origin is None:
            raise ImportError("Could not find the 'trl' library installation path.")

        trl_dir = os.path.dirname(spec.origin)
        dest_trainer_path = os.path.join(trl_dir, "trainer", "grpo_trainer.py")
        dest_config_path = os.path.join(trl_dir, "trainer", "grpo_config.py")
        dest_ivo_trainer_path = os.path.join(trl_dir, "trainer", "ivo_trainer.py")
        dest_ivo_config_path = os.path.join(trl_dir, "trainer", "ivo_config.py")

        source_trainer_path = os.path.join("trainer", "grpo_trainer.py")
        source_config_path = os.path.join("trainer", "grpo_config.py")
        source_ivo_trainer_path = os.path.join("trainer", "ivo_trainer.py")
        source_ivo_config_path = os.path.join("trainer", "ivo_config.py")

        if not os.path.exists(source_trainer_path):
            raise FileNotFoundError(f"Source trainer file not found: {source_trainer_path}")
        if not os.path.exists(source_config_path):
            raise FileNotFoundError(f"Source config file not found: {source_config_path}")
        if not os.path.exists(source_ivo_trainer_path):
            raise FileNotFoundError(f"Source trainer file not found: {source_ivo_trainer_path}")
        if not os.path.exists(source_ivo_config_path):
            raise FileNotFoundError(f"Source config file not found: {source_ivo_config_path}")

        print("Patching TRL with local GRPO/IVO implementation...")
        print(f"  - Copying {source_trainer_path} to {dest_trainer_path}")
        shutil.copyfile(source_trainer_path, dest_trainer_path)
        print(f"  - Copying {source_config_path} to {dest_config_path}")
        shutil.copyfile(source_config_path, dest_config_path)
        print(f"  - Copying {source_ivo_trainer_path} to {dest_ivo_trainer_path}")
        shutil.copyfile(source_ivo_trainer_path, dest_ivo_trainer_path)
        print(f"  - Copying {source_ivo_config_path} to {dest_ivo_config_path}")
        shutil.copyfile(source_ivo_config_path, dest_ivo_config_path)

        _register_ivo_in_trl(trl_dir)
        print("TRL patching complete.")

    except (ImportError, FileNotFoundError, Exception) as e:
        print(f"Error during TRL patching: {e}")
        raise


def _register_ivo_in_trl(trl_dir):
    trainer_init_path = os.path.join(trl_dir, "trainer", "__init__.py")
    trl_init_path = os.path.join(trl_dir, "__init__.py")

    _patch_trl_trainer_init(trainer_init_path)
    _patch_trl_init(trl_init_path)


def _patch_trl_trainer_init(path):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    updated = False
    if '"ivo_trainer": ["IVOTrainer"]' not in text:
        text = text.replace(
            '"grpo_trainer": ["GRPOTrainer"],',
            '"grpo_trainer": ["GRPOTrainer"],\n    "ivo_config": ["IVOConfig"],\n    "ivo_trainer": ["IVOTrainer"],',
        )
        updated = True

    if "from .ivo_trainer import IVOTrainer" not in text:
        text = text.replace(
            "from .grpo_trainer import GRPOTrainer",
            "from .grpo_trainer import GRPOTrainer\n    from .ivo_trainer import IVOTrainer",
        )
        updated = True

    if "from .ivo_config import IVOConfig" not in text:
        text = text.replace(
            "from .grpo_config import GRPOConfig",
            "from .grpo_config import GRPOConfig\n    from .ivo_config import IVOConfig",
        )
        updated = True

    if updated:
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)


def _patch_trl_init(path):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    updated = False
    if '"IVOTrainer"' not in text:
        text = text.replace(
            '"GRPOTrainer",',
            '"GRPOTrainer",\n        "IVOConfig",\n        "IVOTrainer",',
        )
        updated = True

    if "from .trainer import (" in text:
        start = text.find("from .trainer import (")
        end = text.find(")", start)
        if end != -1:
            block = text[start:end]
            if "IVOTrainer" not in block:
                block = block.replace(
                    "GRPOTrainer,",
                    "GRPOTrainer,\n        IVOConfig,\n        IVOTrainer,",
                    1,
                )
                text = text[:start] + block + text[end:]
                updated = True

    if updated:
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
