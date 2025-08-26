"""
This file is adapted from trl.trainers.__init__.py (trl version 0.14.0)
"""

from typing import TYPE_CHECKING

from trl.import_utils import OptionalDependencyNotAvailable, _LazyModule


_import_structure = {
    "grpo_trainer": ["GRPOTrainer"],
    "grpo_config": ["GRPOConfig"],
    "smcgrpo_trainer": ["SMCGRPOTrainer"],
    "smcgrpo_config": ["SMCGRPOConfig"],
}

if TYPE_CHECKING:
    from .grpo_trainer import GRPOTrainer
    from .grpo_config import GRPOConfig
    from .smcgrpo_trainer import SMCGRPOTrainer
    from .smcgrpo_config import SMCGRPOConfig
else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
