# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Optional, Union

import torch
import torch.nn as nn
from transformers import PretrainedConfig

from vllm.config.lora import LoRAConfig
from vllm.model_executor.layers.linear import ReplicatedLinear

from .base_linear import BaseLinearLayerWithLoRA


class ReplicatedLinearWithLoRA(BaseLinearLayerWithLoRA):

    def __init__(self, base_layer: ReplicatedLinear) -> None:
        super().__init__(base_layer, )
        # To ensure interface compatibility, set to 1 always.
        self.output_size = self.base_layer.output_size
        self.n_slices = 1

    def forward(
        self,
        input_: torch.Tensor,
        *,
        lora_input: Optional[torch.Tensor] = None,
        return_separate: bool = False,
    ) -> Union[torch.Tensor,
               tuple[torch.Tensor, torch.Tensor],
               tuple[torch.Tensor, Optional[torch.Tensor]],
               tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]]:
        """Forward of ReplicatedLinearWithLoRA

        Args:
            input_: Tensor whose last dimension is `input_size`.

        Returns:
            - output
            - bias
        """
        bias = (self.base_layer.bias
                if not self.base_layer.skip_bias_add else None)

        # Matrix multiply.
        result = self.apply(input_,
                            bias,
                            lora_x=lora_input,
                            return_separate=return_separate)

        if return_separate:
            output, lora_output = result
        else:
            output = result
            lora_output = None

        output_bias = (self.base_layer.bias
                       if self.base_layer.skip_bias_add else None)

        if not self.base_layer.return_bias:
            return (output, lora_output) if return_separate else output

        return (output, lora_output,
                output_bias) if return_separate else (output, output_bias)

    # ReplicatedLinear should always be replaced, regardless of the fully
    # sharded LoRAs setting, because it is, by definition, copied per GPU.
    @classmethod
    def can_replace_layer(
        cls,
        source_layer: nn.Module,
        lora_config: LoRAConfig,
        packed_modules_list: list,
        model_config: Optional[PretrainedConfig],
    ) -> bool:
        return type(source_layer) is ReplicatedLinear
