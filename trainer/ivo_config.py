from dataclasses import dataclass, field

from .grpo_config import GRPOConfig


@dataclass
class IVOConfig(GRPOConfig):
    """
    Configuration class for the IVOTrainer.

    Extends GRPOConfig with IVO-specific parameters.
    """

    ivo_beta: float = field(
        default=1.0,
        metadata={"help": "Soft label temperature for IVO."},
    )
    normalized_softlabel: bool = field(
        default=True,
        metadata={"help": "Use softmax-normalized soft labels."},
    )
