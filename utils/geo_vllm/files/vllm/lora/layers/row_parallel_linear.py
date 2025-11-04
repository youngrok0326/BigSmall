# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Optional, Union, cast

import torch
import torch.nn as nn
from transformers import PretrainedConfig

from vllm.config.lora import LoRAConfig
from vllm.distributed import (split_tensor_along_last_dim,
                              tensor_model_parallel_all_reduce)
# yapf: disable
from vllm.model_executor.layers.linear import RowParallelLinear
from vllm.platforms import current_platform

from .base_linear import BaseLinearLayerWithLoRA
from .utils import _fully_sharded_can_replace, _not_fully_sharded_can_replace


class RowParallelLinearWithLoRA(BaseLinearLayerWithLoRA):

    def __init__(self, base_layer: RowParallelLinear) -> None:
        super().__init__(base_layer)

        # reset input_size
        self.input_size = self.base_layer.input_size_per_partition
        self.output_size = self.base_layer.output_size
        # There is only one LoRA layer.
        self.n_slices = 1

    def slice_lora_a(self, lora_a: torch.Tensor) -> torch.Tensor:

        shard_size = self.input_size
        start_idx = self.tp_rank * shard_size
        end_idx = (self.tp_rank + 1) * shard_size
        lora_a = lora_a[:,start_idx:end_idx]
        return lora_a

    def slice_lora_b(self, lora_b: torch.Tensor) -> torch.Tensor:
        return lora_b

    def slice_bias(self, bias: torch.Tensor) -> torch.Tensor:
        return bias

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
        """Forward of RowParallelLinear

        Args:
            input_: tensor whose last dimension is `input_size`. If
                    `input_is_parallel` is set, then the last dimension
                    is `input_size // tp_size`.

        Returns:
            - output
            - bias
        """
        # set up backprop all-reduce.
        if self.base_layer.input_is_parallel:
            input_parallel = input_
            lora_parallel = (lora_input if lora_input is not None else
                             input_parallel)
        else:
            # TODO: simplify code below
            splitted_input = split_tensor_along_last_dim(
                input_, num_partitions=self.tp_size)
            input_parallel = splitted_input[self.tp_rank].contiguous()
            if lora_input is None:
                lora_parallel = input_parallel
            else:
                splitted_lora = split_tensor_along_last_dim(
                    lora_input, num_partitions=self.tp_size)
                lora_parallel = splitted_lora[self.tp_rank].contiguous()

        result = self.apply(input_parallel,
                            lora_x=lora_parallel if
                            (return_separate or lora_input is not None) else
                            None,
                            return_separate=return_separate)
        if return_separate:
            output_parallel, lora_output_parallel = result
        else:
            output_parallel = result
            lora_output_parallel = None

        if self.base_layer.reduce_results and self.tp_size > 1:
            output_ = tensor_model_parallel_all_reduce(output_parallel)
            lora_output_ = (tensor_model_parallel_all_reduce(
                lora_output_parallel)
                            if lora_output_parallel is not None else None)
        else:
            output_ = output_parallel
            lora_output_ = lora_output_parallel

        if not self.base_layer.skip_bias_add:
            output = (output_ + self.base_layer.bias
                      if self.base_layer.bias is not None else output_)
            lora_output = (lora_output_ + self.base_layer.bias
                           if (return_separate and lora_output_ is not None
                               and self.base_layer.bias is not None) else
                           lora_output_)
            output_bias = None
        else:
            output = output_
            lora_output = lora_output_
            output_bias = self.base_layer.bias

        if not self.base_layer.return_bias:
            return (output, lora_output) if return_separate else output

        return (output, lora_output,
                output_bias) if return_separate else (output, output_bias)

    @classmethod
    @_not_fully_sharded_can_replace
    def can_replace_layer(
        cls,
        source_layer: nn.Module,
        lora_config: LoRAConfig,
        packed_modules_list: list,
        model_config: Optional[PretrainedConfig],
    ) -> bool:
        return type(source_layer) is RowParallelLinear



# The following layer is based on the tensor parallelism strategy given in
# Y. Sheng et al., S-LoRA: Serving Thousands of Concurrent LoRA Adapters. 2023,
# https://arxiv.org/abs/2311.03285.

class RowParallelLinearWithShardedLoRA(RowParallelLinearWithLoRA):
    """
    Differs from RowParallelLinearWithLoRA by slicing the
    LoRA B's also.

    Based on S-LoRA, slicing happens along the output dim.
    This yields a combined partial sum from the row parallel base
    layer and column partitioned output from the LoRA.
    """

    def slice_lora_b(self, lora_b: torch.Tensor) -> torch.Tensor:
        shard_size = self.lora_b_stacked[0].shape[2]
        start_idx = self.tp_rank * shard_size
        end_idx = (self.tp_rank + 1) * shard_size
        lora_b = lora_b[ start_idx:end_idx,:]
        return lora_b

    def slice_bias(self, bias: torch.Tensor) -> torch.Tensor:
        if bias is None:
            return bias
        self.lora_bias_stacked = cast(tuple[torch.Tensor, ...],
                                      self.lora_bias_stacked)
        shard_size = self.lora_bias_stacked[0].shape[2]
        start_idx = self.tp_rank * shard_size
        end_idx = (self.tp_rank + 1) * shard_size
        bias = bias[start_idx:end_idx]
        return bias

    def apply(
        self,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        *,
        lora_x: Optional[torch.Tensor] = None,
        return_separate: bool = False,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        if lora_x is None and not return_separate:
            output = self.base_layer.quant_method.apply(self.base_layer, x)

            x = x.view(-1, x.shape[-1])
            output, out_orig_shape = output.view(
                -1, output.shape[-1]), output.shape
            buffer = torch.zeros(
                (self.n_slices, x.shape[0], self.lora_a_stacked[0].shape[2]),
                dtype=torch.float32,
                device=x.device,
            )

            shrunk_buffer: Optional[
                torch.Tensor] = self.punica_wrapper.add_shrink(
                    buffer, x, self.lora_a_stacked, 1.0)
            if not current_platform.can_update_inplace():
                buffer = shrunk_buffer
            if self.tp_size > 1:
                buffer = tensor_model_parallel_all_reduce(buffer)

            shard_size = self.lora_b_stacked[0].shape[2]
            offset_start = self.tp_rank * shard_size
            lora_output: Optional[
                torch.Tensor] = self.punica_wrapper.add_expand(
                    output,
                    buffer,
                    self.lora_b_stacked,
                    self.lora_bias_stacked,
                    self.output_slices,
                    offset_start=offset_start,
                    add_input=True,
                )

            if not current_platform.can_update_inplace():
                output = lora_output

            return output.view(*out_orig_shape)

        lora_x = x if lora_x is None else lora_x

        base_output = self.base_layer.quant_method.apply(self.base_layer, x)
        base_shape = base_output.shape
        base_flat = base_output.view(-1, base_output.shape[-1])

        if lora_x is x:
            lora_flat = base_flat.clone()
            lora_shape = base_shape
        else:
            lora_output = self.base_layer.quant_method.apply(
                self.base_layer, lora_x)
            lora_shape = lora_output.shape
            lora_flat = lora_output.view(-1, lora_output.shape[-1])

        base_input_flat = x.view(-1, x.shape[-1])
        lora_input_flat = lora_x.view(-1, lora_x.shape[-1])

        base_buffer = torch.zeros(
            (self.n_slices, base_input_flat.shape[0],
             self.lora_a_stacked[0].shape[2]),
            dtype=torch.float32,
            device=base_input_flat.device,
        )
        base_shrunk = self.punica_wrapper.add_shrink(
            base_buffer, base_input_flat, self.lora_a_stacked, 1.0)
        if not current_platform.can_update_inplace():
            base_buffer = base_shrunk
        if self.tp_size > 1:
            base_buffer = tensor_model_parallel_all_reduce(base_buffer)

        lora_buffer = torch.zeros_like(base_buffer)
        lora_shrunk = self.punica_wrapper.add_shrink(
            lora_buffer, lora_input_flat, self.lora_a_stacked, 1.0)
        if not current_platform.can_update_inplace():
            lora_buffer = lora_shrunk
        if self.tp_size > 1:
            lora_buffer = tensor_model_parallel_all_reduce(lora_buffer)

        shard_size = self.lora_b_stacked[0].shape[2]
        offset_start = self.tp_rank * shard_size
        base_updated = self.punica_wrapper.add_expand(
            base_flat,
            base_buffer,
            self.lora_b_stacked,
            self.lora_bias_stacked,
            self.output_slices,
            offset_start=offset_start,
            add_input=True,
        )
        if not current_platform.can_update_inplace():
            base_flat = base_updated

        lora_updated = self.punica_wrapper.add_expand(
            lora_flat,
            lora_buffer,
            self.lora_b_stacked,
            self.lora_bias_stacked,
            self.output_slices,
            offset_start=offset_start,
            add_input=True,
        )
        if not current_platform.can_update_inplace():
            lora_flat = lora_updated

        return (base_flat.view(*base_shape), lora_flat.view(*lora_shape))

    @classmethod
    @_fully_sharded_can_replace
    def can_replace_layer(
        cls,
        source_layer: nn.Module,
        lora_config: LoRAConfig,
        packed_modules_list: list,
        model_config: Optional[PretrainedConfig],
    ) -> bool:
        # specifying kwargs so they can be easily accessed in decorator
        return super().can_replace_layer(
            source_layer=source_layer,
            lora_config=lora_config,
            packed_modules_list=packed_modules_list,
            model_config=model_config,
            decorate=False,
        )
