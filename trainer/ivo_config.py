from dataclasses import dataclass, field
from typing import Optional

from .grpo_config import GRPOConfig


@dataclass
class IVOConfig(GRPOConfig):
    """
    Configuration class for the IVOTrainer.

    Extends GRPOConfig with IVO-specific parameters.
    """

    beta: float = field(
        default=1.0,
        metadata={"help": "IVO temperature (alpha in paper) used for soft labels and teacher scaling."},
    )
    normalized_softlabel: bool = field(
        default=True,
        metadata={"help": "Use softmax-normalized soft labels."},
    )
    teacher_model: Optional[str] = field(
        default=None,
        metadata={"help": "Teacher model name or path for IVO distillation."},
    )
    teacher_value_beta: Optional[float] = field(
        default=None,
        metadata={"help": "Temperature for teacher value logsumexp. Defaults to beta."},
    )
    alpha: float = field(
        default=0.0,
        metadata={"help": "Teacher guidance strength (beta in paper). Set > 0 to enable distillation."},
    )
    gamma: float = field(
        default=1.0,
        metadata={
            "help": "Geometric mixture weight for old vs teacher log-probs. 1.0=old only, 0.0=teacher only."
        },
    )
    teacher_device: Optional[str] = field(
        default=None,
        metadata={"help": "Device for the teacher model, e.g. 'cuda:0'."},
    )
    teacher_lora_path: Optional[str] = field(
        default=None,
        metadata={"help": "Optional LoRA adapter path or repo for the teacher model."},
    )
    use_value_adapter: bool = field(
        default=False,
        metadata={"help": "Use a value adapter on teacher hidden states for value calibration."},
    )
