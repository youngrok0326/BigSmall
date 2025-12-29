"""
This file is adapted from trl.trainers.__init__.py (trl version 0.14.0)
"""

from typing import TYPE_CHECKING

from trl.import_utils import OptionalDependencyNotAvailable, _LazyModule


_import_structure = {
    "grpo_trainer": ["GRPOTrainer"],
    "grpo_config": ["GRPOConfig"],
    "ivo_trainer": ["IVOTrainer"],
    "ivo_config": ["IVOConfig"],
}

if TYPE_CHECKING:
    from .grpo_trainer import GRPOTrainer
    from .grpo_config import GRPOConfig
    from .ivo_trainer import IVOTrainer
    from .ivo_config import IVOConfig
else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
