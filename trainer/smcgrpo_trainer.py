# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import inspect
import math
import os
import re
import textwrap
import warnings
from collections import defaultdict, deque
from dataclasses import dataclass
from enum import Enum
from collections.abc import Sequence, Sized
from contextlib import nullcontext
from functools import partial
from pathlib import Path
from typing import Any, Callable, Optional, Union

import datasets
import torch
import torch.utils.data
import transformers
from accelerate.utils import broadcast_object_list, gather, gather_object, is_peft_model, set_seed
from datasets import Dataset, IterableDataset
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.utils.data import DataLoader, Sampler
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoProcessor,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    Trainer,
    TrainerCallback,
    is_wandb_available,
)
from transformers.trainer_utils import seed_worker
from transformers.utils import is_datasets_available, is_flash_attn_2_available, is_peft_available, is_rich_available

from trl.data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template
from trl.extras.profiling import profiling_context, profiling_decorator
from trl.extras.vllm_client import VLLMClient
from trl.import_utils import is_liger_kernel_available, is_vllm_available
from trl.models import prepare_deepspeed, prepare_fsdp, unwrap_model_for_generation
from trl.models.utils import _ForwardRedirection
from trl.trainer.callbacks import SyncRefModelCallback
from .grpo_config import GRPOConfig
from trl.trainer.utils import (
    disable_dropout_in_model,
    entropy_from_logits,
    generate_model_card,
    get_comet_experiment_url,
    pad,
    print_prompt_completions_sample,
    selective_log_softmax,
)


if is_peft_available():
    from peft import PeftConfig, get_peft_model

if is_liger_kernel_available():
    from liger_kernel.chunked_loss import LigerFusedLinearGRPOLoss

if is_vllm_available():
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import GuidedDecodingParams
    from trainer.vllm_smc import StepGeneration, SMCVLLM, build_prm_model

if is_wandb_available():
    import wandb

# What we call a reward function is a callable that takes a list of prompts and completions and returns a list of
# rewards. When it's a string, it's a model ID, so it's loaded as a pretrained model.
RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]


class _SMCTrajectoryKind(str, Enum):
    LEGACY = "legacy"
    DROPPED = "dropped"


@dataclass(slots=True)
class _SMCBatchMeta:
    group_counts: list[int]
    group_indices: list[int]
    drop_steps: list[int]
    total_steps: list[int]
    kinds: list[_SMCTrajectoryKind]
    confidences: list[float]

    @property
    def num_groups(self) -> int:
        return len(self.group_counts)

    @property
    def num_samples(self) -> int:
        return len(self.group_indices)


def _smc_is_bundle(payload: Any) -> bool:
    return isinstance(payload, dict) and "completions" in payload and "group_sizes" in payload


def _smc_build_uniform_meta(
    num_samples: int,
    num_generations: int,
    fallback_steps: int,
    confidences: Optional[Sequence[float]] = None,
) -> _SMCBatchMeta:
    if num_generations <= 0:
        raise ValueError("num_generations must be positive")
    if num_samples % num_generations != 0:
        raise ValueError(
            f"Batch size {num_samples} must be divisible by num_generations {num_generations}."
        )
    num_groups = num_samples // num_generations
    group_counts = [num_generations] * num_groups
    group_indices = [group for group in range(num_groups) for _ in range(num_generations)]
    drop_steps = [-1] * len(group_indices)
    total_steps = [fallback_steps] * len(group_indices)
    kinds = [_SMCTrajectoryKind.LEGACY] * len(group_indices)
    if confidences is None:
        confidences = [1.0 for _ in range(num_samples)]
    if len(confidences) != num_samples:
        raise ValueError("confidences length must match num_samples")
    return _SMCBatchMeta(
        group_counts,
        group_indices,
        drop_steps,
        total_steps,
        kinds,
        list(map(float, confidences)),
    )


def _smc_flatten_bundle(
    payload: dict[str, Any],
    fallback_steps: int,
    limit_per_group: Optional[int] = None,
) -> tuple[list[list[int]], _SMCBatchMeta]:
    completions_by_group: list[list[list[int]]] = payload.get("completions", [])  # type: ignore[assignment]
    completion_confidences: list[list[float]] = payload.get("completion_confidences", [])  # type: ignore[assignment]
    saved: list[list[tuple[int, list[int]]]] = payload.get("saved", [])  # type: ignore[assignment]
    saved_confidences: list[list[float]] = payload.get("saved_confidences", [])  # type: ignore[assignment]
    reasoning_steps: list[int] = payload.get("group_reasoning_steps", [])  # type: ignore[assignment]

    flattened: list[list[int]] = []
    group_counts: list[int] = []
    group_indices: list[int] = []
    drop_steps: list[int] = []
    total_steps: list[int] = []
    kinds: list[_SMCTrajectoryKind] = []
    confidences: list[float] = []

    if limit_per_group is not None and limit_per_group < 0:
        limit_per_group = 0

    def _sample_extras(
        extras: list[tuple[int, list[int]]],
        confs: list[float],
        limit: Optional[int],
    ) -> tuple[list[tuple[int, list[int]]], list[float]]:
        if not extras or limit is None or limit >= len(extras):
            return extras, confs
        if limit <= 0:
            return [], []

        by_step: dict[int, list[int]] = {}
        for idx, (step_idx, _seq) in enumerate(extras):
            step_val = int(step_idx)
            by_step.setdefault(step_val, []).append(idx)

        total = sum(len(indices) for indices in by_step.values())
        if total <= limit:
            return extras

        allocations: dict[int, int] = {}
        fractional: list[tuple[float, int]] = []
        for step_val, indices in by_step.items():
            raw = len(indices) * limit / total
            assign = min(len(indices), int(math.floor(raw)))
            allocations[step_val] = assign
            fractional.append((raw - assign, step_val))

        assigned = sum(allocations.values())
        remaining = limit - assigned
        if remaining > 0:
            fractional.sort(key=lambda item: (-item[0], item[1]))
            for _frac, step_val in fractional:
                if remaining == 0:
                    break
                capacity = len(by_step[step_val]) - allocations[step_val]
                if capacity <= 0:
                    continue
                take = min(capacity, remaining)
                allocations[step_val] += take
                remaining -= take
        if remaining > 0:
            for step_val in sorted(by_step.keys()):
                if remaining == 0:
                    break
                capacity = len(by_step[step_val]) - allocations[step_val]
                if capacity <= 0:
                    continue
                take = min(capacity, remaining)
                allocations[step_val] += take
                remaining -= take

        selected_indices: list[int] = []
        for step_val, indices in by_step.items():
            take = allocations.get(step_val, 0)
            if take <= 0:
                continue
            perm = torch.randperm(len(indices))
            chosen = perm[:take].tolist()
            selected_indices.extend(indices[idx] for idx in chosen)

        selected_indices.sort()
        return [extras[idx] for idx in selected_indices], [confs[idx] for idx in selected_indices]

    for group_idx, primary in enumerate(completions_by_group):
        extras = saved[group_idx] if group_idx < len(saved) else []
        extras_conf = saved_confidences[group_idx] if group_idx < len(saved_confidences) else []
        extras, extras_conf = _sample_extras(extras, extras_conf, limit_per_group)
        if len(extras_conf) < len(extras):
            extras_conf = extras_conf + [1.0] * (len(extras) - len(extras_conf))
        step_cap = int(reasoning_steps[group_idx]) if group_idx < len(reasoning_steps) else fallback_steps
        primary_conf = []
        if completion_confidences:
            if group_idx < len(completion_confidences):
                primary_conf = completion_confidences[group_idx]
            else:
                primary_conf = []

        for seq_idx, seq in enumerate(primary):
            flattened.append(seq)
            group_indices.append(group_idx)
            drop_steps.append(-1)
            total_steps.append(step_cap)
            kinds.append(_SMCTrajectoryKind.LEGACY)
            if primary_conf and seq_idx < len(primary_conf):
                confidences.append(float(primary_conf[seq_idx]))
            else:
                confidences.append(1.0)

        for (step_idx, seq), conf in zip(extras, extras_conf):
            flattened.append(seq)
            group_indices.append(group_idx)
            drop_steps.append(int(step_idx))
            total_steps.append(max(step_cap, int(step_idx) + 1))
            kinds.append(_SMCTrajectoryKind.DROPPED)
            confidences.append(float(conf))

        group_counts.append(len(primary) + len(extras))

    meta = _SMCBatchMeta(group_counts, group_indices, drop_steps, total_steps, kinds, confidences)
    return flattened, meta


def _smc_expand_batch(
    inputs: Sequence[dict[str, Any]],
    prompts: Sequence[Any],
    original_prompts: Sequence[Any],
    prompts_text: Sequence[str],
    images: Optional[Sequence[Any]],
    num_generations: int,
    meta: _SMCBatchMeta,
) -> tuple[list[dict[str, Any]], list[Any], list[Any], list[str], Optional[list[Any]]]:
    if num_generations <= 0:
        raise ValueError("num_generations must be positive")
    if len(inputs) % num_generations != 0:
        raise ValueError("SMC bundle size mismatch: batch is not a multiple of num_generations")

    num_groups = len(inputs) // num_generations
    base_inputs = [copy.deepcopy(inputs[g * num_generations]) for g in range(num_groups)]
    base_prompts = [copy.deepcopy(prompts[g * num_generations]) for g in range(num_groups)]
    base_original_prompts = [copy.deepcopy(original_prompts[g * num_generations]) for g in range(num_groups)]
    base_prompts_text = [prompts_text[g * num_generations] for g in range(num_groups)]
    base_images = None if images is None else [images[g * num_generations] for g in range(num_groups)]

    expanded_inputs: list[dict[str, Any]] = []
    expanded_prompts: list[Any] = []
    expanded_original_prompts: list[Any] = []
    expanded_prompts_text: list[str] = []
    expanded_images: Optional[list[Any]] = [] if base_images is not None else None

    for group_idx, count in enumerate(meta.group_counts):
        for _ in range(count):
            expanded_inputs.append(copy.deepcopy(base_inputs[group_idx]))
            expanded_prompts.append(copy.deepcopy(base_prompts[group_idx]))
            expanded_original_prompts.append(copy.deepcopy(base_original_prompts[group_idx]))
            expanded_prompts_text.append(base_prompts_text[group_idx])
            if expanded_images is not None and base_images is not None:
                expanded_images.append(base_images[group_idx])

    return expanded_inputs, expanded_prompts, expanded_original_prompts, expanded_prompts_text, expanded_images


def _smc_drop_bonus(
    kinds: Sequence[_SMCTrajectoryKind],
    drop_steps: Sequence[int],
    total_steps: Sequence[int],
    scheme: str,
    device: torch.device,
) -> torch.Tensor:
    reward = torch.zeros(len(kinds), dtype=torch.float32, device=device)
    if not kinds:
        return reward
    drop_mask = torch.tensor([kind == _SMCTrajectoryKind.DROPPED for kind in kinds], dtype=torch.bool, device=device)
    if not drop_mask.any():
        return reward
    drop_tensor = torch.tensor(drop_steps, dtype=torch.float32, device=device)
    total_tensor = torch.tensor(total_steps, dtype=torch.float32, device=device)
    denom = torch.where(total_tensor <= 0, torch.ones_like(total_tensor), total_tensor)
    progress = torch.clamp(drop_tensor[drop_mask] + 1.0, min=0.0) / denom[drop_mask]
    scheme = scheme.lower()
    if scheme == "penalty":
        reward[drop_mask] = -(1.0 - progress)
    else:
        # Default to proportional progress
        reward[drop_mask] = progress
    return reward


def _max_variance_subset(rewards: torch.Tensor, subset_size: int) -> list[int]:
    if subset_size <= 0 or rewards.numel() == 0:
        return []
    if rewards.numel() <= subset_size:
        return list(range(rewards.numel()))
    if subset_size == 1:
        return [int(torch.argmax(rewards).item())]

    sorted_vals, sorted_indices = torch.sort(rewards, descending=True)
    sorted_indices = sorted_indices.tolist()

    def _variance(idx_list: list[int]) -> float:
        if not idx_list:
            return float("-inf")
        selected = rewards[idx_list]
        return float(torch.var(selected, unbiased=False).item())

    best_indices = sorted_indices[:subset_size]
    best_variance = _variance(best_indices)

    for i in range(subset_size + 1):
        prefix = sorted_indices[:i]
        suffix_count = subset_size - i
        if suffix_count < 0:
            continue
        suffix = sorted_indices[-suffix_count:] if suffix_count > 0 else []
        candidate = prefix + suffix
        if len(candidate) != subset_size:
            continue
        candidate_variance = _variance(candidate)
        if candidate_variance > best_variance:
            best_variance = candidate_variance
            best_indices = candidate

    return best_indices


def _build_legacy_keep_mask(
    group_indices: Sequence[int],
    kinds: Sequence[str],
    rewards: torch.Tensor,
    num_generations_grad: int,
) -> list[bool]:
    rewards = rewards.detach()
    total = len(group_indices)
    keep_mask: list[bool] = [False] * total
    if total == 0:
        return keep_mask

    dropped_label = _SMCTrajectoryKind.DROPPED.value
    legacy_label = _SMCTrajectoryKind.LEGACY.value

    groups: dict[int, list[int]] = defaultdict(list)
    for idx, (group_idx, kind) in enumerate(zip(group_indices, kinds)):
        groups[int(group_idx)].append(idx)
        if kind == dropped_label:
            keep_mask[idx] = True
        elif kind not in {legacy_label, dropped_label}:
            keep_mask[idx] = True

    for _, indices in groups.items():
        legacy_indices = [idx for idx in indices if kinds[idx] == legacy_label]
        if not legacy_indices:
            continue
        if len(legacy_indices) <= num_generations_grad:
            for pos in legacy_indices:
                keep_mask[pos] = True
            continue
        legacy_rewards = rewards[legacy_indices]
        selected_rel = _max_variance_subset(legacy_rewards, num_generations_grad)
        for rel_idx in selected_rel:
            keep_mask[legacy_indices[rel_idx]] = True

    return keep_mask

class RepeatSampler(Sampler):
    """
    Sampler that repeats the indices of a dataset in a structured manner.

    Args:
        data_source (`Sized`):
            Dataset to sample from.
        mini_repeat_count (`int`):
            Number of times to repeat each index per batch.
        batch_size (`int`, *optional*, defaults to `1`):
            Number of unique indices per batch.
        repeat_count (`int`, *optional*, defaults to `1`):
            Number of times to repeat the full sampling process.
        shuffle (`bool`, *optional*, defaults to `True`):
            Whether to shuffle the dataset.
        seed (`int` or `None`, *optional*, defaults to `None`):
            Random seed for reproducibility (only affects this sampler).

    Example:
    ```python
    >>> sampler = RepeatSampler(
    ...     ["a", "b", "c", "d", "e", "f", "g"], mini_repeat_count=2, batch_size=3, repeat_count=4
    ... )
    >>> list(sampler)
    [4, 4, 3, 3, 0, 0,
     4, 4, 3, 3, 0, 0,
     4, 4, 3, 3, 0, 0,
     4, 4, 3, 3, 0, 0,
     1, 1, 2, 2, 6, 6,
     1, 1, 2, 2, 6, 6,
     1, 1, 2, 2, 6, 6,
     1, 1, 2, 2, 6, 6]
    ```

    ```txt
    mini_repeat_count = 3
          -   -   -
         [0,  0,  0,  1,  1,  1,  2,  2,  2,  3,  3,  3,      |
          4,  4,  4,  5,  5,  5,  6,  6,  6,  7,  7,  7,      |
          8,  8,  8,  9,  9,  9, 10, 10, 10, 11, 11, 11,      |
                                                                repeat_count = 2
          0,  0,  0,  1,  1,  1,  2,  2,  2,  3,  3,  3,      |
          4,  4,  4,  5,  5,  5,  6,  6,  6,  7,  7,  7,      |
          8,  8,  8,  9,  9,  9, 10, 10, 10, 11, 11, 11, ...] |
          ---------   ---------   ---------   ---------
           ---------   ---------   ---------   ---------
            ---------   ---------   ---------   ---------
                         batch_size = 12
    ```
    """

    def __init__(
        self,
        data_source: Sized,
        mini_repeat_count: int,
        batch_size: int = 1,
        repeat_count: int = 1,
        shuffle: bool = True,
        seed: Optional[int] = None,
    ):
        self.data_source = data_source
        self.mini_repeat_count = mini_repeat_count
        self.batch_size = batch_size
        self.repeat_count = repeat_count
        self.num_samples = len(data_source)
        self.shuffle = shuffle
        self.seed = seed

        if shuffle:
            self.generator = torch.Generator()  # Create a local random generator
            if seed is not None:
                self.generator.manual_seed(seed)

    def __iter__(self):
        if self.shuffle:
            # E.g., [2, 4, 3, 1, 0, 6, 5] (num_samples = 7)
            indexes = torch.randperm(self.num_samples, generator=self.generator).tolist()
        else:
            indexes = list(range(self.num_samples))

        #    [2, 4, 3, 1, 0, 6, 5]
        # -> [[2, 4, 3], [1, 0, 6], [5]]  (batch_size = 3)
        indexes = [indexes[i : i + self.batch_size] for i in range(0, len(indexes), self.batch_size)]

        #    [[2, 4, 3], [1, 0, 6], [5]]
        # -> [[2, 4, 3], [1, 0, 6]]
        indexes = [chunk for chunk in indexes if len(chunk) == self.batch_size]

        for chunk in indexes:
            for _ in range(self.repeat_count):
                for index in chunk:
                    for _ in range(self.mini_repeat_count):
                        yield index

    def __len__(self) -> int:
        return (self.num_samples // self.batch_size) * self.batch_size * self.mini_repeat_count * self.repeat_count


# torch.nanstd doesn't exist, so we define it here
def nanstd(tensor: torch.Tensor) -> torch.Tensor:
    """
    Compute the standard deviation of a tensor, ignoring NaNs. This function only supports 1D tensors.

    Args:
        tensor (`torch.Tensor`):
            Input tensor of shape `(N,)`.

    Returns:
        `torch.Tensor`:
            Standard deviation of the tensor, ignoring NaNs.
    """
    variance = torch.nanmean((tensor - torch.nanmean(tensor, keepdim=True)) ** 2)  # Compute variance ignoring NaNs
    count = torch.sum(~torch.isnan(tensor))  # Count of non-NaN values
    variance *= count / (count - 1)  # Bessel's correction
    return torch.sqrt(variance)


def split_tensor_dict(
    tensor_dict: dict[str, Optional[torch.Tensor]], num_chunks: int
) -> list[dict[str, Optional[torch.Tensor]]]:
    """
    Splits a dictionary of tensors along the first dimension into `num_chunks` equal parts.

    Example:
    ```python
    >>> x = torch.arange(12).reshape(6, 2)
    >>> y = torch.arange(6).reshape(6, 1)
    >>> tensor_dict = {"x": x, "y": y}
    >>> split_tensor_dict(tensor_dict, 3)
    [
        {"x": tensor([[0, 1], [2, 3]]), "y": tensor([[0], [1]])},
        {"x": tensor([[4, 5], [6, 7]]), "y": tensor([[2], [3]])},
        {"x": tensor([[ 8,  9], [10, 11]]), "y": tensor([[4], [5]])}
    ]
    ```
    """
    first_tensor = next(tensor for tensor in tensor_dict.values() if tensor is not None)
    chunk_size = first_tensor.shape[0] // num_chunks
    return [
        {
            key: tensor[i * chunk_size : (i + 1) * chunk_size] if tensor is not None else None
            for key, tensor in tensor_dict.items()
        }
        for i in range(num_chunks)
    ]


def shuffle_sequence_dict(seq_dict: dict[str, Optional[Sequence]]) -> dict[str, Optional[Sequence]]:
    """
    Shuffles all sequence-like values in a dictionary along the first dimension in unison.

    Example:
    ```python
    >>> x = torch.arange(6).reshape(3, 2)
    >>> y = ["a", "b", "c"]
    >>> seq_dict = {"x": x, "y": y}
    >>> shuffle_sequence_dict(seq_dict)
    {'x': tensor([[2, 3],
                  [0, 1],
                  [4, 5]]),
     'y': ['b', 'a', 'c']}
    ```
    """
    # Determine batch size from the first non-None sequence
    batch_size = len(next(v for v in seq_dict.values() if v is not None))
    permutation = torch.randperm(batch_size)

    def permute(v: Optional[Sequence]) -> Optional[Sequence]:
        if v is None:
            return None
        if isinstance(v, torch.Tensor):
            return v[permutation]
        return [v[i] for i in permutation]

    return {key: permute(val) for key, val in seq_dict.items()}


def nanmin(tensor: torch.Tensor) -> torch.Tensor:
    """
    Compute the minimum value of a tensor, ignoring NaNs. This function only supports 1D tensors.

    Args:
        tensor (`torch.Tensor`): Input tensor of shape `(N,)`.

    Returns:
        `torch.Tensor`: Minimum value of the tensor, ignoring NaNs. Returns NaN if all values are NaN.
    """
    if torch.isnan(tensor).all():
        return torch.tensor(float("nan"), dtype=tensor.dtype, device=tensor.device)
    return torch.min(tensor[~torch.isnan(tensor)])


def nanmax(tensor: torch.Tensor) -> torch.Tensor:
    """
    Compute the maximum value of a tensor, ignoring NaNs. This function only supports 1D tensors.

    Args:
        tensor (`torch.Tensor`): Input tensor of shape `(N,)`.

    Returns:
        `torch.Tensor`: Maximum value of the tensor, ignoring NaNs. Returns NaN if all values are NaN.
    """
    if torch.isnan(tensor).all():
        return torch.tensor(float("nan"), dtype=tensor.dtype, device=tensor.device)
    return torch.max(tensor[~torch.isnan(tensor)])


def identity(x):
    """Do we really need docs for this?"""
    return x


def split_pixel_values_by_grid(batch: dict[str, torch.Tensor]) -> dict[str, Union[torch.Tensor, list[torch.Tensor]]]:
    """
    Splits `batch["pixel_values"]` into a list of tensors based on the product of each row in
    `batch["image_grid_thw"]`, while keeping other entries unchanged.
    """
    if "image_grid_thw" not in batch or "pixel_values" not in batch:
        return batch

    lengths = batch["image_grid_thw"].prod(dim=1).tolist()  # [batch_size]
    pixel_values = batch["pixel_values"]  # [total, feature_dim]

    if sum(lengths) != pixel_values.size(0):
        raise ValueError(f"Mismatch: sum(lengths) = {sum(lengths)} != pixel_values.size(0) = {pixel_values.size(0)}")

    split_values = list(torch.split(batch["pixel_values"], lengths, dim=0))
    return {**batch, "pixel_values": split_values}


def unsplit_pixel_values_by_grid(batch: dict[str, Union[torch.Tensor, list[torch.Tensor]]]) -> dict[str, torch.Tensor]:
    """
    Opposite of `split_pixel_values_by_grid`. Merges a list of tensors in `batch["pixel_values"]`
    back into a single tensor along the first dimension.
    """
    pixel_values = batch.get("pixel_values")

    if isinstance(pixel_values, list):
        merged = torch.cat(pixel_values, dim=0)
        return {**batch, "pixel_values": merged}
    else:
        return batch


def truncate_with_protected_tokens(
    ids: torch.Tensor, mask: torch.Tensor, target_length: int, protected_tokens: list[int]
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Truncate tensors to target length while preserving protected tokens.

    Args:
        ids (`torch.Tensor`):
            Input tensor of token IDs, shape (batch_size, sequence_length).
        mask (`torch.Tensor`):
            Input tensor of attention masks, shape (batch_size, sequence_length).
        target_length (`int`):
            Desired length of the output sequences.
        protected_tokens (`list[int]`):
            List of token IDs that should be preserved in the output.
    """
    protected_set = set(protected_tokens)

    def process_sequence(ids, mask):
        # Create boolean masks
        is_protected = torch.tensor([x.item() in protected_set for x in ids])
        is_non_protected = ~is_protected

        # Count tokens
        num_protected = is_protected.sum().item()
        num_non_protected_needed = target_length - num_protected

        if num_non_protected_needed < 0:
            raise ValueError(
                f"target_length ({target_length}) is too small for the protected tokens ({num_protected} tokens). "
                f"Please increase target length to at least {num_protected} or disable truncation."
            )

        # Select which non-protected tokens to keep (rightmost ones)
        non_protected_indices = torch.where(is_non_protected)[0]
        keep_non_protected = torch.zeros_like(is_non_protected)
        if num_non_protected_needed > 0:
            keep_indices = non_protected_indices[-num_non_protected_needed:]
            keep_non_protected[keep_indices] = True

        # Final mask: protected OR selected non-protected
        keep_mask = is_protected | keep_non_protected

        return ids[keep_mask], mask[keep_mask]

    # Process each sequence in the batch
    truncated_seq = []
    truncated_mask = []

    for i in range(ids.shape[0]):
        new_ids, new_mask = process_sequence(ids[i], mask[i])
        truncated_seq.append(new_ids)
        truncated_mask.append(new_mask)

    return torch.stack(truncated_seq), torch.stack(truncated_mask)


class GRPOTrainer(Trainer):
    """
    Trainer for the Group Relative Policy Optimization (GRPO) method. This algorithm was initially proposed in the
    paper [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language
    Models](https://huggingface.co/papers/2402.03300).

    Example:

    ```python
    from datasets import load_dataset
    from trl import GRPOTrainer

    dataset = load_dataset("trl-lib/tldr", split="train")


    def reward_func(completions, **kwargs):
        # Dummy reward function that rewards completions with more unique letters.
        return [float(len(set(completion))) for completion in completions]


    trainer = GRPOTrainer(
        model="Qwen/Qwen2-0.5B-Instruct",
        reward_funcs=reward_func,
        train_dataset=dataset,
    )

    trainer.train()
    ```

    Args:
        model (`Union[str, PreTrainedModel]`):
            Model to be trained. Can be either:

            - A string, being the *model id* of a pretrained model hosted inside a model repo on huggingface.co, or a
              path to a *directory* containing model weights saved using
              [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is loaded
              using [`~transformers.AutoModelForCausalLM.from_pretrained`] with the keyword arguments in
              `args.model_init_kwargs`.
            - A [`~transformers.PreTrainedModel`] object. Only causal language models are supported.
        reward_funcs (`Union[RewardFunc, list[RewardFunc]]`):
            Reward functions to be used for computing the rewards. To compute the rewards, we call all the reward
            functions with the prompts and completions and sum the rewards. Can be either:

            - A single reward function, such as:
                - A string: The *model ID* of a pretrained model hosted inside a model repo on huggingface.co, or a
                path to a *directory* containing model weights saved using
                [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is loaded
                using [`~transformers.AutoModelForSequenceClassification.from_pretrained`] with `num_labels=1` and the
                keyword arguments in `args.model_init_kwargs`.
                - A [`~transformers.PreTrainedModel`] object: Only sequence classification models are supported.
                - A custom reward function: The function is provided with the prompts and the generated completions,
                  plus any additional columns in the dataset. It should return a list of rewards. Custom reward
                  functions can also return `None` when the reward is not applicable to those samples. This is useful
                  for multi-task training where different reward functions apply to different types of samples. When a
                  reward function returns `None` for a sample, that reward function is excluded from the reward
                  calculation for that sample. For more details, see [Using a custom reward
                  function](#using-a-custom-reward-function).

                  The trainer's state is also passed to the reward function. The trainer's state is an instance of
                  [`~transformers.TrainerState`] and can be accessed by accessing the `trainer_state` argument to the
                  reward function's signature.
            - A list of reward functions, where each item can independently be any of the above types. Mixing different
            types within the list (e.g., a string model ID and a custom reward function) is allowed.
        args ([`GRPOConfig`], *optional*, defaults to `None`):
            Configuration for this trainer. If `None`, a default configuration is used.
        train_dataset ([`~datasets.Dataset`] or [`~datasets.IterableDataset`]):
            Dataset to use for training. It must include a column `"prompt"`. Any additional columns in the dataset is
            ignored. The format of the samples can be either:

            - [Standard](dataset_formats#standard): Each sample contains plain text.
            - [Conversational](dataset_formats#conversational): Each sample contains structured messages (e.g., role
              and content).
        eval_dataset ([`~datasets.Dataset`], [`~datasets.IterableDataset`] or `dict[str, Union[Dataset, IterableDataset]]`):
            Dataset to use for evaluation. It must meet the same requirements as `train_dataset`.
        processing_class ([`~transformers.PreTrainedTokenizerBase`] or [`~transformers.ProcessorMixin`], *optional*, defaults to `None`):
            Processing class used to process the data. The padding side must be set to "left". If `None`, the
            processing class is loaded from the model's name with [`~transformers.AutoProcessor.from_pretrained`]. A
            padding token, `tokenizer.pad_token`, must be set. If the processing class has not set a padding token,
            `tokenizer.eos_token` will be used as the default.
        reward_processing_classes (`Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]`, *optional*, defaults to `None`):
            Processing classes corresponding to the reward functions specified in `reward_funcs`. Can be either:

            - A single processing class: Used when `reward_funcs` contains only one reward function.
            - A list of processing classes: Must match the order and length of the reward functions in `reward_funcs`.
            If set to `None`, or if an element of the list corresponding to a [`~transformers.PreTrainedModel`] is
            `None`, the tokenizer for the model is automatically loaded using
            [`~transformers.AutoTokenizer.from_pretrained`]. For elements in `reward_funcs` that are custom reward
            functions (not [`~transformers.PreTrainedModel`]), the corresponding entries in `reward_processing_classes`
            are ignored.
        callbacks (list of [`~transformers.TrainerCallback`], *optional*, defaults to `None`):
            List of callbacks to customize the training loop. Will add those to the list of default callbacks detailed
            in [here](https://huggingface.co/docs/transformers/main_classes/callback).

            If you want to remove one of the default callbacks used, use the [`~transformers.Trainer.remove_callback`]
            method.
        optimizers (`tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`, *optional*, defaults to `(None, None)`):
            A tuple containing the optimizer and the scheduler to use. Will default to an instance of [`AdamW`] on your
            model and a scheduler given by [`get_linear_schedule_with_warmup`] controlled by `args`.
        peft_config ([`~peft.PeftConfig`], *optional*, defaults to `None`):
            PEFT configuration used to wrap the model. If `None`, the model is not wrapped.
    """

    _tag_names = ["trl", "grpo"]

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        args: Optional[GRPOConfig] = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]] = None,
        processing_class: Optional[Union[PreTrainedTokenizerBase, ProcessorMixin]] = None,
        reward_processing_classes: Optional[Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        peft_config: Optional["PeftConfig"] = None,
    ):
        # Args
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = GRPOConfig(f"{model_name}-GRPO")

        # Models
        # Trained model
        model_init_kwargs = args.model_init_kwargs or {}
        if isinstance(model, str):
            model_id = model
            torch_dtype = model_init_kwargs.get("torch_dtype")
            if isinstance(torch_dtype, torch.dtype) or torch_dtype == "auto" or torch_dtype is None:
                pass  # torch_dtype is already a torch.dtype or "auto" or None
            elif isinstance(torch_dtype, str):  # it's a str, but not "auto"
                torch_dtype = getattr(torch, torch_dtype)
                model_init_kwargs["torch_dtype"] = torch_dtype
            else:
                raise ValueError(
                    "Invalid `torch_dtype` passed to `GRPOConfig`. Expected either 'auto' or a string representing "
                    f"a `torch.dtype` (e.g., 'float32'), but got {torch_dtype}."
                )
            # Disable caching if gradient checkpointing is enabled (not supported)
            config = AutoConfig.from_pretrained(model_id)
            architecture = getattr(transformers, config.architectures[0])
            model = architecture.from_pretrained(model_id, **model_init_kwargs)
        else:
            model_id = model.config._name_or_path
            if args.model_init_kwargs is not None:
                raise ValueError(
                    "You passed `model_init_kwargs` to the `GRPOConfig`, but your model is already instantiated. "
                    "This argument can only be used when the `model` argument is a string."
                )

        # Some models (SmolVLM/Idefics3) don't support `logits_to_keep` argument and error out if we pass it
        # Inspect the forward method before we wrap the model with PEFT
        self.model_kwarg_keys = (
            inspect.signature(model.forward).parameters.keys()
            if not hasattr(model, "get_base_model")
            else inspect.signature(model.get_base_model().forward).parameters.keys()
        )

        if peft_config is not None:
            if not is_peft_available():
                raise ImportError("PEFT is required to use `peft_config`. Run `pip install peft`.")
            model = get_peft_model(model, peft_config)

        # Enable gradient checkpointing if requested
        if args.gradient_checkpointing:
            model = self._enable_gradient_checkpointing(model, args)

        # Processing class
        if processing_class is None:
            processing_class = AutoProcessor.from_pretrained(model.config._name_or_path)

        # Handle pad token for processors or tokenizers
        if isinstance(processing_class, ProcessorMixin):
            tokenizer = processing_class.tokenizer
        elif isinstance(processing_class, PreTrainedTokenizerBase):
            tokenizer = processing_class
        else:
            raise TypeError("The `processing_class` must be either a `PreTrainedTokenizerBase` or a `ProcessorMixin`")

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        self.pad_token = tokenizer.pad_token
        self.pad_token_id = tokenizer.pad_token_id
        self.eos_token_id = tokenizer.eos_token_id
        self.image_token = getattr(processing_class, "image_token", None)
        self.image_token_id = getattr(processing_class, "image_token_id", None)
        self.vision_start_token_id = getattr(model.config, "vision_start_token_id", None)
        self.vision_end_token_id = getattr(model.config, "vision_end_token_id", None)

        # Reward functions
        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]
        self.reward_func_names = []
        for i, reward_func in enumerate(reward_funcs):
            if isinstance(reward_func, str):
                reward_funcs[i] = AutoModelForSequenceClassification.from_pretrained(
                    reward_func, num_labels=1, **model_init_kwargs
                )
            if isinstance(reward_funcs[i], nn.Module):  # Use Module over PretrainedModel for compat w/ compiled models
                self.reward_func_names.append(reward_funcs[i].config._name_or_path.split("/")[-1])
            else:
                self.reward_func_names.append(reward_funcs[i].__name__)
        self.reward_funcs = reward_funcs

        # Reward weights
        if args.reward_weights is not None:
            if len(args.reward_weights) != len(reward_funcs):
                raise ValueError(
                    f"Number of reward weights ({len(args.reward_weights)}) must match number of reward "
                    f"functions ({len(reward_funcs)})"
                )
            self.reward_weights = torch.tensor(args.reward_weights, dtype=torch.float32)
        else:
            self.reward_weights = torch.ones(len(reward_funcs), dtype=torch.float32)

        # Reward processing class
        if reward_processing_classes is None:
            reward_processing_classes = [None] * len(reward_funcs)
        elif not isinstance(reward_processing_classes, list):
            reward_processing_classes = [reward_processing_classes]
        else:
            if len(reward_processing_classes) != len(reward_funcs):
                raise ValueError("The number of reward processing classes must match the number of reward functions.")

        for i, (reward_processing_class, reward_func) in enumerate(zip(reward_processing_classes, reward_funcs)):
            if isinstance(reward_func, PreTrainedModel):
                if reward_processing_class is None:
                    reward_processing_class = AutoTokenizer.from_pretrained(reward_func.config._name_or_path)
                if reward_processing_class.pad_token_id is None:
                    reward_processing_class.pad_token = reward_processing_class.eos_token
                # The reward model computes the reward for the latest non-padded token in the input sequence.
                # So it's important to set the pad token ID to the padding token ID of the processing class.
                reward_func.config.pad_token_id = reward_processing_class.pad_token_id
                reward_processing_classes[i] = reward_processing_class
        self.reward_processing_classes = reward_processing_classes

        # Training arguments
        self.max_prompt_length = args.max_prompt_length
        self.max_completion_length = args.max_completion_length  # = |o_i| in the GRPO paper
        self.num_generations = args.num_generations  # = G in the GRPO paper
        self.num_generations_grad = args.num_generations_grad
        self.num_generations_grad_raw = args.num_generations_grad
        if self.num_generations_grad is None:
            self.num_generations_grad = self.num_generations
        self._downsample_disabled = (
            self.num_generations_grad_raw is not None and self.num_generations_grad_raw <= 0
        )
        self._num_generations_grad_effective = (
            self.num_generations if self._downsample_disabled else self.num_generations_grad
        )
        if self._downsample_disabled:
            self.num_generations_grad = self._num_generations_grad_effective
        self._smc_drop_coef = 0.1
        self._smc_drop_reward_scheme = "progress"
        self._smc_return_all_limit: Optional[int] = None
        self._smc_return_eos: bool = False
        self._smc_self_reward: bool = False
        self._enable_drop_bonus = False
        self.temperature = args.temperature
        self.top_p = args.top_p
        self.top_k = args.top_k
        self.min_p = args.min_p
        self.repetition_penalty = args.repetition_penalty
        self.use_transformers_paged = args.use_transformers_paged
        self.use_vllm = args.use_vllm
        self.vllm_mode = args.vllm_mode
        self.vllm_gpu_memory_utilization = args.vllm_gpu_memory_utilization  # only applies to colocation mode
        self.vllm_tensor_parallel_size = args.vllm_tensor_parallel_size  # only applies to colocation mode
        self.use_liger_loss = args.use_liger_loss
        self.loss_type = args.loss_type
        self.scale_rewards = args.scale_rewards
        self.importance_sampling_level = args.importance_sampling_level
        self.mask_truncated_completions = args.mask_truncated_completions
        self.top_entropy_quantile = args.top_entropy_quantile
        if self.use_liger_loss and self.top_entropy_quantile < 1.0:
            raise NotImplementedError(
                "Liger Kernels don't currently support masking token positions based on entropy."
            )
        if self.use_liger_loss and not self.importance_sampling_level == "token":
            raise NotImplementedError(
                "Liger Kernels currently only support token-level importance sampling. Please set"
                "`importance_sampling_level` to 'token'."
            )

        # Datasets
        self.shuffle_dataset = args.shuffle_dataset

        if (
            isinstance(train_dataset, IterableDataset)
            or isinstance(eval_dataset, IterableDataset)
            or (
                isinstance(eval_dataset, dict) and any(isinstance(ds, IterableDataset) for ds in eval_dataset.values())
            )
        ):
            # See https://github.com/huggingface/trl/issues/3213
            raise NotImplementedError(
                "Iterable datasets are not yet supported in GRPOTrainer. Please use a standard dataset instead."
            )

        # Multi-step
        self.num_iterations = args.num_iterations  # = 𝜇 in the GRPO paper
        self.epsilon_low = args.epsilon
        self.epsilon_high = args.epsilon_high if args.epsilon_high is not None else args.epsilon
        # Tracks the number of iterations (forward + backward passes), including those within a grad accum cycle
        self._step = 0
        # Buffer the batch to reuse generated outputs across multiple updates. For more details, see
        # `_get_train_sampler` and `_prepare_inputs`.
        self._buffered_inputs = None

        # The trainer estimates the number of FLOPs (floating-point operations) using the number of elements in the
        # input tensor associated with the key "input_ids". However, in GRPO, the sampled data does not include the
        # "input_ids" key. Instead, the available keys is "prompt". As a result, the trainer issues the warning:
        # "Could not estimate the number of tokens of the input, floating-point operations will not be computed." To
        # suppress this warning, we set the "estimate_tokens" key in the model's "warnings_issued" dictionary to True.
        # This acts as a flag to indicate that the warning has already been issued.
        model.warnings_issued["estimate_tokens"] = True

        super().__init__(
            model=model,
            args=args,
            data_collator=identity,  # No data collation is needed in GRPO
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
        )

        # Reference model
        self.beta = args.beta
        if self.beta == 0.0:
            # If beta is 0.0, the reference model is not needed
            self.ref_model = None
        elif is_peft_model(model):
            # If PEFT is used, the reference model is not needed since the adapter can be disabled
            # to revert to the initial model.
            self.ref_model = None
        else:
            # For deepspeed, fsdp or non-distributed models, create a reference model from scratch
            config = AutoConfig.from_pretrained(model_id)
            architecture = getattr(transformers, config.architectures[0])
            self.ref_model = architecture.from_pretrained(model_id, **model_init_kwargs)

        # Disable dropout in the models
        if args.disable_dropout:
            disable_dropout_in_model(model)
            if self.ref_model is not None:
                disable_dropout_in_model(self.ref_model)

        # Liger loss
        if self.use_liger_loss:
            if not is_liger_kernel_available():
                raise ImportError(
                    "Liger is required to use `liger_loss` as the GRPO loss. Run `pip install liger-kernel`."
                )
            # redirect the model.module forward to the model forward to ensure pre-forward hooks are called
            self._forward_redirection = _ForwardRedirection()

            self.liger_grpo_loss = LigerFusedLinearGRPOLoss(
                beta=self.beta,
                epsilon_low=self.epsilon_low,
                epsilon_high=self.epsilon_high,
                temperature=self.temperature,
                use_ref_model=self.beta != 0.0,
                loss_type=self.loss_type,
                max_completion_length=self.max_completion_length,
            )

        # Initialize the metrics
        self._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
        self._total_train_tokens = 0
        self.log_completions = args.log_completions
        self.wandb_log_unique_prompts = args.wandb_log_unique_prompts
        self.num_completions_to_print = args.num_completions_to_print
        # Keep logs sized to the generation batch to record only outputs from the latest model update.
        self._logs = {
            "image": deque(maxlen=args.generation_batch_size),
            "prompt": deque(maxlen=args.generation_batch_size),
            "completion": deque(maxlen=args.generation_batch_size),
            "rewards": defaultdict(lambda: deque(maxlen=args.generation_batch_size)),
            "advantages": deque(maxlen=args.generation_batch_size),
            "self_confidence": deque(maxlen=args.generation_batch_size),
            "self_weighted_reward": deque(maxlen=args.generation_batch_size),
        }

        # Ensure each process receives a unique seed to prevent duplicate completions when generating with
        # transformers if num_generations exceeds per_device_train_batch_size. We could skip it if we use vLLM, but
        # it's safer to set it in all cases.
        set_seed(args.seed, device_specific=True)

        if self.use_vllm:
            if not is_vllm_available():
                raise ImportError(
                    "vLLM is not available and `use_vllm` is set to True. Please install vLLM with "
                    "`pip install vllm` to use it."
                )

            if self.vllm_mode == "server" and self.accelerator.is_main_process:
                if args.vllm_server_base_url is not None:
                    base_url = args.vllm_server_base_url
                else:
                    base_url = f"http://{args.vllm_server_host}:{args.vllm_server_port}"
                self.vllm_client = VLLMClient(base_url=base_url, connection_timeout=args.vllm_server_timeout)
                self.vllm_client.init_communicator(device=torch.cuda.current_device())

            elif self.vllm_mode == "colocate":
                # Make sure vllm_tensor_parallel_size group size evenly divides the world size - each group should have
                # the same number of ranks
                if not self.accelerator.num_processes % self.vllm_tensor_parallel_size == 0:
                    raise ValueError(
                        f"vllm_tensor_parallel_size ({self.vllm_tensor_parallel_size}) must divide world size "
                        f"({self.accelerator.num_processes}) evenly."
                    )

                if self.vllm_tensor_parallel_size > 1:
                    # Create subgroups of ranks for TP, each group with `vllm_tensor_parallel_size` ranks.
                    # For example, if world_size=8 and vllm_tensor_parallel_size=2 → groups: [0,1], [2,3], [4,5], [6,7]
                    self.tp_group, _ = torch.distributed.new_subgroups_by_enumeration(
                        [
                            list(range(i * self.vllm_tensor_parallel_size, (i + 1) * self.vllm_tensor_parallel_size))
                            for i in range(self.accelerator.num_processes // self.vllm_tensor_parallel_size)
                        ]
                    )

                # vLLM requires the environment variables to be set for distributed training.
                os.environ["RANK"] = str(self.accelerator.process_index)
                os.environ["LOCAL_RANK"] = str(self.accelerator.local_process_index)
                os.environ["WORLD_SIZE"] = str(self.accelerator.num_processes)
                os.environ["MASTER_ADDR"] = os.environ.get("MASTER_ADDR", "localhost")
                os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "12345")

                if self.max_prompt_length is not None and self.max_completion_length is not None:
                    max_model_len = self.max_prompt_length + self.max_completion_length
                else:
                    max_model_len = None
                self.llm = LLM(
                    model=model.name_or_path,
                    tensor_parallel_size=args.vllm_tensor_parallel_size,
                    gpu_memory_utilization=self.vllm_gpu_memory_utilization,
                    max_num_seqs=self.args.per_device_train_batch_size
                    * self.vllm_tensor_parallel_size
                    * self.args.steps_per_generation,
                    max_model_len=max_model_len,
                    distributed_executor_backend="external_launcher",
                    # Feed identical seed for tp groups to ensure sampling results are the same across workers
                    seed=self.accelerator.process_index // self.vllm_tensor_parallel_size,
                    # Latest vLLM v1 memory profiler is misled by the high default value (i.e., 32768) - thinking there's not enough memory
                    max_num_batched_tokens=4096,
                    model_impl=self.args.vllm_model_impl,
                )

            # vLLM specific sampling arguments
            self.guided_decoding_regex = args.vllm_guided_decoding_regex

            self._last_loaded_step = -1  # tag to avoid useless loading during grad accumulation

            # When using vLLM, the main process is responsible for loading the model weights. This can cause process
            # desynchronization and seems to lead to DeepSpeed hanging during initialization. To prevent this, we
            # synchronize all processes after vLLM has been fully initialized.
            self.accelerator.wait_for_everyone()
        else:
            generation_kwargs = {
                "max_new_tokens": self.max_completion_length,
                "do_sample": True,
                "pad_token_id": tokenizer.pad_token_id,
                "bos_token_id": tokenizer.bos_token_id,
                "eos_token_id": tokenizer.eos_token_id,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "top_k": self.top_k,
                "min_p": self.min_p,
                "repetition_penalty": self.repetition_penalty,
                "cache_implementation": args.cache_implementation,
            }
            if args.use_transformers_paged:
                generation_kwargs["max_batch_tokens"] = 512
                generation_kwargs["num_blocks"] = 1024
                generation_kwargs["block_size"] = 128
            if args.generation_kwargs is not None:
                generation_kwargs.update(args.generation_kwargs)
            self.generation_config = GenerationConfig(**generation_kwargs)
            
            # Pull SMC params from generation_kwargs if provided
            _smc_cfg = {}
            if getattr(self.args, "generation_kwargs", None):
                _smc_cfg = (self.args.generation_kwargs or {}).get("smc", {})
            return_all_cfg = bool(_smc_cfg.get("return_all", False))
            self._smc_return_eos = bool(_smc_cfg.get("return_eos", self._smc_return_eos))
            self._smc_self_reward = bool(_smc_cfg.get("self_reward", self._smc_self_reward))
            drop_scheme = str(_smc_cfg.get("drop_reward_scheme", self._smc_drop_reward_scheme)).lower()
            if drop_scheme not in {"progress", "penalty"}:
                drop_scheme = "progress"
            self._smc_drop_reward_scheme = drop_scheme
            limit_cfg = _smc_cfg.get("return_all_limit_per_group", self._smc_return_all_limit)
            if limit_cfg is None:
                self._smc_return_all_limit = None
            else:
                self._smc_return_all_limit = max(int(limit_cfg), 0)
            self._smc_drop_coef = float(_smc_cfg.get("drop_reward_coef", self._smc_drop_coef))
            self._set_drop_bonus_enabled(return_all_cfg)

            smc_params = {
                "use_smc": bool(_smc_cfg),
                "num_generations": self.args.num_generations,
                "smc_beta": _smc_cfg.get("smc_beta", 1.0),
                "smc_warmup_tokens": getattr(self.args, "smc_warmup_tokens", 0),
                "smc_confidence_eta": _smc_cfg.get("smc_confidence_eta", 1.0),
                "smc_resample_threshold": getattr(self.args, "smc_resample_threshold", 0.5),
                "smc_confidence_window_size": _smc_cfg.get("smc_confidence_window_size", 50),
                "smc_topk": _smc_cfg.get("smc_topk", -1),
            }
            for key, value in smc_params.items():
                setattr(self.generation_config, key, value)

        # Gradient accumulation requires scaled loss. Normally, loss scaling in the parent class depends on whether the
        # model accepts loss-related kwargs. Since we compute our own loss, this check is irrelevant. We set
        # self.model_accepts_loss_kwargs to False to enable scaling.
        self.model_accepts_loss_kwargs = False

        # Add tags to the model
        self.model.add_model_tags(self._tag_names)

        if self.ref_model is not None:
            if self.is_deepspeed_enabled:
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            elif self.is_fsdp_enabled:
                self.ref_model = prepare_fsdp(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

        if args.sync_ref_model:
            self.add_callback(SyncRefModelCallback(ref_model=self.ref_model, accelerator=self.accelerator))

        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                if self.is_deepspeed_enabled:
                    self.reward_funcs[i] = prepare_deepspeed(reward_func, self.accelerator)
                else:
                    # set device placement to True to make `prepare_model` move `reward_func` to device when using fsdp
                    self.reward_funcs[i] = self.accelerator.prepare_model(
                        reward_func, evaluation_mode=True, device_placement=True
                    )
                    

    def _set_signature_columns_if_needed(self):
        # If `self.args.remove_unused_columns` is True, non-signature columns are removed.
        # By default, this method sets `self._signature_columns` to the model's expected inputs.
        # In GRPOTrainer, we preprocess data, so using the model's signature columns doesn't work.
        # Instead, we set them to the columns expected by the `training_step` method, hence the override.
        if self._signature_columns is None:
            self._signature_columns = ["prompt", "image"]

    # This method overrides `Trainer.get_train_dataloader` to support our custom batching strategy.
    # Instead of returning a standard per-step batch (i.e., `per_device_batch_size), our dataloader loads an
    # *generation* batch (i.e., `per_device_batch_size × steps_per_generation`). This allows us to generate completions
    # once every steps_per_generation step—rather than once per accumulation step—which is significantly more
    # efficient. The only change from the original implementation is multiplying the batch size by
    # `steps_per_generation`. Thus, `_prepare_inputs` is called with this *generation* batch, and it handles the
    # splitting internally.
    # Maintenance note: This method is a copy-paste of the original `Trainer.get_train_dataloader` with only one line
    # modification. As a result, some parts of the method aren't relevant to GRPO, but we keep them to stay one line
    # apart from the super method, ensuring easier maintenance in the future.
    def get_train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

        dataloader_params = {
            "batch_size": self._train_batch_size * self.args.steps_per_generation,  # < this is the change
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_train_sampler()
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = partial(
                seed_worker, num_workers=self.args.dataloader_num_workers, rank=self.args.process_index
            )

            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))

    def _get_train_sampler(self, dataset: Optional[Dataset] = None) -> Sampler:
        # Returns a sampler that
        # 1. ensures each prompt is repeated across multiple processes. This guarantees that identical prompts are
        #    distributed to different GPUs, allowing rewards to be computed and normalized correctly within each prompt
        #    group. Using the same seed across processes ensures consistent prompt assignment, preventing discrepancies
        #    in group formation.
        # 2. repeats the batch multiple times to allow reusing generations across multiple updates. Refer to
        #    _prepare_inputs to see how the generations are stored and reused.

        # In the following figure, the values are the prompt indices. The first row shows the first sampled batch, the
        # second row shows the second sampled batch, and so on.
        #
        #                                      |   GPU 0  |   GPU 1  |
        #
        #                 global_step   step    <-───>  num_generations=2
        #                                       <-───────> per_device_train_batch_size=3
        #  grad_accum    ▲  ▲  0          0     0   0   1   1   2   2   <- Generate for the first `steps_per_generation` (prompts 0 to 11); store the completions; use the first slice to compute the loss
        #     =2         ▼  |  0          1     3   3   4   4   5   5   <- Take the stored generations and use the second slice to compute the loss
        #                   |
        #                   |  1          2     6   6   7   7   8   8   <- Take the stored generations and use the third slice to compute the loss
        #  steps_per_gen=4  ▼  1          3     9   9  10  10  11  11   <- Take the stored generations and use the fourth slice to compute the loss
        #
        #                      2          4    12  12  13  13  14  14   <- Generate for the second `steps_per_generation` (prompts 12 to 23); store the completions; use the first slice to compute the loss
        #                      2          5    15  15  16  16  17  17   <- Take the stored generations and use the second slice to compute the loss
        #                                          ...
        if dataset is None:
            dataset = self.train_dataset
        return RepeatSampler(
            data_source=dataset,
            mini_repeat_count=self.num_generations,
            batch_size=self.args.generation_batch_size // self.num_generations,
            repeat_count=self.num_iterations * self.args.steps_per_generation,
            shuffle=self.shuffle_dataset,
            seed=self.args.seed,
        )

    def _get_eval_sampler(self, eval_dataset) -> Sampler:
        # See _get_train_sampler for an explanation of the sampler.
        return RepeatSampler(
            data_source=eval_dataset,
            mini_repeat_count=self.num_generations,
            seed=self.args.seed,
        )

    def _enable_gradient_checkpointing(self, model: PreTrainedModel, args: GRPOConfig) -> PreTrainedModel:
        """Enables gradient checkpointing for the model."""
        # Ensure use_cache is disabled
        model.config.use_cache = False

        # Enable gradient checkpointing on the base model for PEFT
        if is_peft_model(model):
            model.base_model.gradient_checkpointing_enable()
        # Enable gradient checkpointing for non-PEFT models
        else:
            model.gradient_checkpointing_enable()

        gradient_checkpointing_kwargs = args.gradient_checkpointing_kwargs or {}
        use_reentrant = (
            "use_reentrant" not in gradient_checkpointing_kwargs or gradient_checkpointing_kwargs["use_reentrant"]
        )

        if use_reentrant:
            model.enable_input_require_grads()

        return model

    @profiling_decorator
    def _get_last_hidden_state(
        self,
        unwrapped_model,
        input_ids,
        attention_mask,
        logits_to_keep,
        pixel_values=None,
        image_grid_thw=None,
        pixel_attention_mask=None,
        image_sizes=None,
    ):
        if is_peft_model(unwrapped_model):
            unwrapped_model = unwrapped_model.base_model.model

        # Build model inputs - check if the model supports logits_to_keep (some models and VLMs don't)
        model_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}

        # For Qwen models:
        if image_grid_thw is not None and pixel_values is not None:
            model_inputs["image_grid_thw"] = image_grid_thw
        # For Gemma, SmolVLM2, LLaVa-Next etc.:
        if pixel_values is not None:
            model_inputs["pixel_values"] = pixel_values
        # For SmolVLM2
        if pixel_attention_mask is not None:
            model_inputs["pixel_attention_mask"] = pixel_attention_mask
        # For LLaVa-Next
        if image_sizes is not None:
            model_inputs["image_sizes"] = image_sizes

        # Only add logits_to_keep if the model supports it
        if "logits_to_keep" in self.model_kwarg_keys:
            # We add 1 to `logits_to_keep` because the last logits of the sequence is later excluded
            model_inputs["logits_to_keep"] = logits_to_keep + 1

        last_hidden_state = unwrapped_model.model(**model_inputs).last_hidden_state
        # Exclude the last value: it corresponds to the next token pred
        last_hidden_state = last_hidden_state[:, :-1, :]  # (B, L-1, H)
        # Only keep the last logits_to_keep. For model that support logits_to_keep, this is a no-op.
        last_hidden_state = last_hidden_state[:, -logits_to_keep:, :]  # (B, logits_to_keep, H)
        return last_hidden_state

    def get_high_entropy_mask(
        self, entropies: torch.Tensor, mask: torch.Tensor, threshold: float, accelerator=None
    ) -> torch.Tensor:
        """
        Returns a binary mask identifying tokens whose entropy exceeds a given quantile threshold.

        Args:
            entropies (`torch.Tensor`):
                Tensor of shape (batch_size, seq_len) with per-token entropy values.
            mask (`torch.Tensor`):
                Binary mask of the same shape as `entropies`, where `1` indicates valid tokens and `0` padding.
            threshold (`float`):
                Quantile threshold between `0.0` and `1.0` to select high-entropy tokens.

        Returns:
            `torch.Tensor`:
                Boolean mask of shape (batch_size, seq_len), where `True` indicates tokens with entropy >= threshold and
                `False` otherwise.
        """
        non_pad_entropies = entropies[mask.bool()].float()
        if non_pad_entropies.numel() == 0:
            return torch.zeros_like(entropies, dtype=torch.bool)
        all_non_pad_entropies = self.accelerator.gather(non_pad_entropies)
        # Filter out any empty tensors that might result from processes with no valid tokens
        entropy_threshold = torch.quantile(all_non_pad_entropies, threshold)
        masked_entropies = entropies * mask.float()
        entropy_mask = masked_entropies >= entropy_threshold
        return entropy_mask & mask.bool()  # ensure padding tokens are always masked out

    @profiling_decorator
    def _get_per_token_logps_and_entropies(
        self,
        model,
        input_ids,
        attention_mask,
        logits_to_keep,
        batch_size=None,
        compute_entropy=False,
        pixel_values=None,
        image_grid_thw=None,
        pixel_attention_mask=None,
        image_sizes=None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], int]:
        """Compute log-probs/entropies and return the effective completion length used."""

        batch_size = batch_size or input_ids.size(0)  # Chunk inputs into smaller batches to reduce memory peak
        all_logps: list[torch.Tensor] = []
        all_entropies: list[torch.Tensor] = []
        effective_keep = max(int(logits_to_keep), 1)

        for start in range(0, input_ids.size(0), batch_size):
            input_ids_batch = input_ids[start : start + batch_size]
            attention_mask_batch = attention_mask[start : start + batch_size]

            # Build model inputs - check if the model supports logits_to_keep (some models and VLMs don't)
            model_inputs = {"input_ids": input_ids_batch, "attention_mask": attention_mask_batch}

            if image_grid_thw is not None and pixel_values is not None:
                model_inputs["image_grid_thw"] = image_grid_thw[start : start + batch_size]
                start_pixel_idx = image_grid_thw[:start].prod(-1).sum().item()
                end_pixel_idx = image_grid_thw[: start + batch_size].prod(-1).sum().item()
                model_inputs["pixel_values"] = pixel_values[start_pixel_idx:end_pixel_idx]
            elif pixel_values is not None:
                model_inputs["pixel_values"] = pixel_values[start : start + batch_size]
            if pixel_attention_mask is not None:
                model_inputs["pixel_attention_mask"] = pixel_attention_mask[start : start + batch_size]
            if image_sizes is not None:
                model_inputs["image_sizes"] = image_sizes[start : start + batch_size]

            # Only add logits_to_keep if the model supports it
            if "logits_to_keep" in self.model_kwarg_keys:
                # We add 1 to `logits_to_keep` because the last logits of the sequence is later excluded
                model_inputs["logits_to_keep"] = logits_to_keep + 1

            logits = model(**model_inputs).logits
            # Exclude the last value: it corresponds to the next token pred
            logits = logits[:, :-1, :]  # (B, L-1, H)

            chunk_keep = min(
                logits_to_keep,
                logits.size(1),
                input_ids_batch.size(1),
            )
            chunk_keep = max(int(chunk_keep), 1)
            effective_keep = min(effective_keep, chunk_keep)

            # Only keep the last chunk_keep tokens. For models that support logits_to_keep, this is a no-op.
            logits = logits[:, -chunk_keep:, :]  # (B, chunk_keep, H)

            # Divide logits by sampling temperature.
            # See https://huggingface.co/blog/the_n_implementation_details_of_rlhf_with_ppo#policy-training-implementation-details
            logits = logits / self.temperature

            completion_ids = input_ids_batch[:, -chunk_keep:]
            logps = selective_log_softmax(logits, completion_ids)  # compute logprobs
            all_logps.append(logps)

            if compute_entropy:
                with torch.no_grad():
                    entropies = entropy_from_logits(logits)
                all_entropies.append(entropies)

        if effective_keep < logits_to_keep:
            all_logps = [logp[:, -effective_keep:] for logp in all_logps]
            if compute_entropy:
                all_entropies = [entropy[:, -effective_keep:] for entropy in all_entropies]

        logps = torch.cat(all_logps, dim=0)
        entropies = torch.cat(all_entropies, dim=0) if compute_entropy else None
        return logps, entropies, effective_keep

    def _fix_param_name_to_vllm(self, name, extra_prefixes: Optional[list[str]] = None):
        extra_prefixes = extra_prefixes or []
        prefixes = ["_checkpoint_wrapped_module."] + extra_prefixes
        for prefix in prefixes:
            name = name.replace(prefix, "")
        return name

    def _sync_fsdp1_params_to_vllm(self, module: nn.Module, prefix: str = "", visited=None):
        """Memory-efficient post-order traversal of FSDP modules to extract full parameters and sync with vLLM."""
        # For FSDP1, we need to recurse into children and also use summon_full_params
        if visited is None:
            visited = set()
        for child_name, child_module in module.named_children():
            child_prefix = f"{prefix}.{child_name}" if prefix else child_name
            self._sync_fsdp1_params_to_vllm(
                child_module, prefix=child_prefix, visited=visited
            )  # recurse into the child

        if isinstance(module, FSDP):
            with FSDP.summon_full_params(module, recurse=False, writeback=False):
                for param_name, param in module.named_parameters():
                    full_name = f"{prefix}.{param_name}" if prefix else param_name
                    full_name = self._fix_param_name_to_vllm(full_name, extra_prefixes=["_fsdp_wrapped_module."])

                    if full_name in visited:
                        continue  # skip FSDP subtrees already traversed
                    visited.add(full_name)

                    if self.vllm_mode == "server" and self.accelerator.is_main_process:
                        self.vllm_client.update_named_param(full_name, param.data)
                    elif self.vllm_mode == "colocate":
                        llm_model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
                        llm_model.load_weights([(full_name, param.data)])

    def _sync_fsdp2_params_to_vllm(self, module: nn.Module):
        # For FSDP2, module.state_dict() already covers all parameters, so no need for recursion
        for name, param in module.state_dict().items():
            if param.is_cpu:
                param = param.to(torch.device("cuda"))
            param = param.full_tensor()

            if self.vllm_mode == "server" and self.accelerator.is_main_process:
                self.vllm_client.update_named_param(name, param)
            elif self.vllm_mode == "colocate":
                llm_model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
                llm_model.load_weights([(name, param)])

    @profiling_decorator
    def _move_model_to_vllm(self):
        # For DeepSpeed ZeRO-3 and FSDP, we need to gather all parameters before operations
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        zero_stage_3 = deepspeed_plugin is not None and deepspeed_plugin.zero_stage == 3
        if zero_stage_3:
            import deepspeed

            gather_if_zero3 = deepspeed.zero.GatheredParameters
        else:
            gather_if_zero3 = nullcontext

        if is_peft_model(self.model):
            # With PEFT and FSDP/DeepSpeed ZeRO Stage 3, we must gather the full model at once before merging, as
            # merging adapters in a sharded manner is not supported.
            # TODO: does this work with FSDP?
            with gather_if_zero3(list(self.model.parameters())):
                self.model.merge_adapter()

                # Update vLLM weights while parameters are gathered
                if self.is_fsdp_enabled:  # note if using FSDP, gather_if_zero3 is nullcontext
                    # Update vLLM weights while parameters are gathered
                    # For PEFT with FSDP we need to use the memory efficient post-order traversal
                    fsdp_plugin = getattr(self.accelerator.state, "fsdp_plugin", None)
                    fsdp_version = getattr(fsdp_plugin, "fsdp_version", 1) if fsdp_plugin else 1
                    if fsdp_version == 1:
                        self._sync_fsdp1_params_to_vllm(
                            self.model
                        )  # use memory-efficient post-order traversal for FSDP
                    elif fsdp_version == 2:
                        self._sync_fsdp2_params_to_vllm(self.model)
                else:
                    # DeepSpeed ZeRO-3 with PEFT
                    for name, param in self.model.named_parameters():
                        # When using PEFT, we need to recover the original parameter name and discard some parameters
                        name = name.removeprefix("base_model.model.").replace(".base_layer", "")
                        if self.model.prefix in name:
                            continue
                        # When module to save, remove its prefix and discard the original module
                        if "original_module" in name:
                            continue
                        name = self._fix_param_name_to_vllm(name, extra_prefixes=["modules_to_save.default."])

                        if self.vllm_mode == "server" and self.accelerator.is_main_process:
                            self.vllm_client.update_named_param(name, param.data)
                        elif self.vllm_mode == "colocate":
                            llm_model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
                            llm_model.load_weights([(name, param.data)])
                # Unmerge adapters while parameters are still gathered
                self.model.unmerge_adapter()
                # Parameters will automatically be repartitioned when exiting the context
        else:
            # For non-PEFT models, simply gather (if needed) and update each parameter individually.
            if self.is_fsdp_enabled:
                fsdp_plugin = getattr(self.accelerator.state, "fsdp_plugin", None)
                fsdp_version = getattr(fsdp_plugin, "fsdp_version", 1) if fsdp_plugin else 1
                if fsdp_version == 1:
                    self._sync_fsdp1_params_to_vllm(self.model)  # use memory-efficient post-order traversal for FSDP
                elif fsdp_version == 2:
                    self._sync_fsdp2_params_to_vllm(self.model)
            else:
                for name, param in self.model.named_parameters():
                    name = self._fix_param_name_to_vllm(name)
                    with gather_if_zero3([param]):
                        if self.vllm_mode == "server" and self.accelerator.is_main_process:
                            self.vllm_client.update_named_param(name, param.data)
                        elif self.vllm_mode == "colocate":
                            llm_model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
                            llm_model.load_weights([(name, param.data)])

        # Reset cache on vLLM
        if self.vllm_mode == "server" and self.accelerator.is_main_process:
            self.vllm_client.reset_prefix_cache()
        elif self.vllm_mode == "colocate":
            self.llm.reset_prefix_cache()

    def _current_drop_reward_name(self) -> str:
        return "drop_penalty" if self._smc_drop_reward_scheme == "penalty" else "drop_progress"

    def _set_drop_bonus_enabled(self, enable: bool) -> None:
        """Synchronize the drop-reward head with the generator configuration and scheme."""
        enable = bool(enable)
        dtype = self.reward_weights.dtype
        device = self.reward_weights.device
        reward_names = list(self.reward_func_names)
        weights_list = [float(w) for w in self.reward_weights.tolist()]

        drop_names = {"drop_progress", "drop_penalty"}
        # Remove any previously registered drop reward to avoid duplicates when scheme changes.
        idx = 0
        while idx < len(reward_names):
            if reward_names[idx] in drop_names:
                reward_names.pop(idx)
                weights_list.pop(idx)
                continue
            idx += 1

        if enable:
            reward_names.append(self._current_drop_reward_name())
            weights_list.append(float(self._smc_drop_coef))

        self.reward_func_names = reward_names
        self.reward_weights = torch.tensor(weights_list, dtype=dtype, device=device)
        self._enable_drop_bonus = enable

    @profiling_decorator
    def _prepare_inputs(
        self, generation_batch: dict[str, Union[torch.Tensor, Any]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        # Prepares inputs for model training/evaluation by managing completion generation and batch handling.
        # During training:
        #   - Receives the local generation batch (Per-GPU batch size × steps per generation)
        #     from the modified training dataloader instead of the standard local batch
        #   - Generates completions once for the entire generation batch and splits it into batches of size
        #     `per_device_train_batch_size`
        #   - Buffers these completions and returns the appropriate slice for the current accumulation step
        #   - Optimizes by regenerating completions only periodically (every steps_per_generation * num_iterations)
        # During evaluation:
        #   - The input is treated as a standard local batch (no accumulation, no multiple iterations)
        #   - Completions are generated for each batch without buffering or reuse
        # Returns a single local batch in both cases.

        mode = "train" if self.model.training else "eval"
        if mode == "train":
            generate_every = self.args.steps_per_generation * self.num_iterations
            if self._step % generate_every == 0 or self._buffered_inputs is None:
                # self._buffered_inputs=None can occur when resuming from a checkpoint
                generation_batch = self._generate_and_score_completions(generation_batch)
                generation_batch = split_pixel_values_by_grid(generation_batch)
                generation_batch = shuffle_sequence_dict(generation_batch)
                generation_batches = split_tensor_dict(generation_batch, self.args.steps_per_generation)
                self._buffered_inputs = [unsplit_pixel_values_by_grid(batch) for batch in generation_batches]
            inputs = self._buffered_inputs[self._step % self.args.steps_per_generation]
            self._step += 1
        else:
            # In evaluation, there is neither batch grouping for generation, nor multiple iterations, hence
            # local generation batch == local eval batch
            inputs = self._generate_and_score_completions(generation_batch)
        return inputs

    @profiling_decorator
    def _calculate_rewards(self, inputs, prompts, completions, completion_ids_list):
        device = self.accelerator.device
        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)

        # Repeat all input columns (but "prompt", "completion", and "completion_ids") to match the num of generations
        keys = [key for key in inputs[0] if key not in ["prompt", "completion", "completion_ids"]]
        reward_kwargs = {key: [example[key] for example in inputs] for key in keys}

        # This allows for dynamic reward shaping based on training progress.
        reward_kwargs["trainer_state"] = self.state

        for i, (reward_func, reward_processing_class, reward_func_name) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes, self.reward_func_names)
        ):
            with profiling_context(self, reward_func_name):
                if isinstance(reward_func, nn.Module):  # Module (no PretrainedModel) for compat with compiled models
                    if is_conversational(inputs[0]):
                        messages = [{"messages": p + c} for p, c in zip(prompts, completions)]
                        texts = [apply_chat_template(x, reward_processing_class)["text"] for x in messages]
                    else:
                        texts = [p + c for p, c in zip(prompts, completions)]
                    reward_inputs = reward_processing_class(
                        text=texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
                    )
                    reward_inputs = super()._prepare_inputs(reward_inputs)
                    with torch.inference_mode():
                        rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]  # Shape (B*G,)
                else:
                    output_reward_func = reward_func(
                        prompts=prompts, completions=completions, completion_ids=completion_ids_list, **reward_kwargs
                    )
                    # Convert None values to NaN
                    output_reward_func = [reward if reward is not None else torch.nan for reward in output_reward_func]

                    rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        # If all reward functions return None for a given row, issue a detailed warning
        if torch.isnan(rewards_per_func).all(dim=1).any():
            nan_row_idx = torch.isnan(rewards_per_func).all(dim=1).nonzero(as_tuple=True)[0][0]
            row_reward_kwargs = {key: value[nan_row_idx] for key, value in reward_kwargs.items()}
            row_reward_kwargs["prompt"] = prompts[nan_row_idx]
            row_reward_kwargs["completion"] = completions[nan_row_idx]
            warnings.warn(
                f"All reward functions returned None for the following kwargs: {row_reward_kwargs}. "
                "Please ensure that at least one reward function returns a valid reward."
            )

        # Gather the reward per function: this part is crucial, because the rewards are normalized per group and the
        # completions may be distributed across processes
        rewards_per_func = gather(rewards_per_func)
        return rewards_per_func

    @torch.no_grad()
    def _generate_completions_smc(self, unwrapped_model: nn.Module, prompt_ids: torch.Tensor, prompt_mask: torch.Tensor) -> dict:
        """Wrapper to call the external TSMC generator function."""
        return self.smc_generator.generate(unwrapped_model, prompt_ids, prompt_mask)

    def _ensure_smc_vllm(self) -> "SMCVLLM":
        smc_cfg: dict[str, Any] = {}
        if getattr(self.args, "generation_kwargs", None):
            smc_cfg = (self.args.generation_kwargs or {}).get("smc", {})
            prm_cfg = (self.args.generation_kwargs or {}).get("prm", {})
        else:
            prm_cfg = {}

        step_token = smc_cfg.get("step_token")
        stop_token = smc_cfg.get("stop_token")
        max_steps = int(smc_cfg.get("max_steps", 32))
        tokens_per_step = smc_cfg.get("tokens_per_step")
        if tokens_per_step is not None:
            tokens_per_step = int(tokens_per_step)
        confidence_cfg = smc_cfg.get("confidence", {})
        conf_scoring = str(confidence_cfg.get("scoring", smc_cfg.get("scoring", "entropy")))
        conf_group = str(confidence_cfg.get("group", "mean"))
        conf_aggregation = str(confidence_cfg.get("aggregation", "last"))
        conf_from_base = bool(confidence_cfg.get("from_base_model", False))
        return_all = bool(smc_cfg.get("return_all", False))
        return_eos = bool(smc_cfg.get("return_eos", False))
        self._smc_return_eos = return_eos
        self._smc_self_reward = bool(smc_cfg.get("self_reward", self._smc_self_reward))
        random_sampling = bool(smc_cfg.get("random_sampling", False))
        smc_topk = int(smc_cfg.get("smc_topk", -1))
        conf_window = int(smc_cfg.get("smc_confidence_window_size", 512))
        conf_eta = float(smc_cfg.get("smc_confidence_eta", 1.0))
        cdf_alpha_val = confidence_cfg.get("cdf_alpha") if confidence_cfg else None
        if cdf_alpha_val is None:
            cdf_alpha_val = smc_cfg.get("cdf_alpha", 0.25)
        cdf_alpha = float(cdf_alpha_val)
        include_stop = smc_cfg.get("include_stop_str_in_output", True)
        drop_scheme = str(smc_cfg.get("drop_reward_scheme", self._smc_drop_reward_scheme)).lower()
        if drop_scheme not in {"progress", "penalty"}:
            drop_scheme = "progress"
        self._smc_drop_reward_scheme = drop_scheme
        limit_cfg = smc_cfg.get("return_all_limit_per_group", self._smc_return_all_limit)
        if limit_cfg is None:
            self._smc_return_all_limit = None
        else:
            self._smc_return_all_limit = max(int(limit_cfg), 0)
        self._smc_drop_coef = float(smc_cfg.get("drop_reward_coef", self._smc_drop_coef))
        self._set_drop_bonus_enabled(return_all)
        report_to = getattr(self.args, "report_to", None)

        if isinstance(report_to, str):
            report_to_list = [report_to]
        elif isinstance(report_to, (list, tuple)):
            report_to_list = list(report_to)
        else:
            report_to_list = []
        log_wandb = ("wandb" in report_to_list) and self.is_world_process_zero()

        prm_signature = None
        prm_model = None
        if prm_cfg and bool(prm_cfg.get("use_prm")):
            prm_signature = tuple(sorted((str(k), repr(v)) for k, v in prm_cfg.items()))
            if getattr(self, "_smc_prm_signature", None) != prm_signature:
                self._smc_prm = build_prm_model(prm_cfg)
                self._smc_prm_signature = prm_signature
            prm_model = getattr(self, "_smc_prm", None)
        else:
            self._smc_prm = None
            self._smc_prm_signature = None

        model_limit = None
        if self.max_prompt_length is not None and self.max_completion_length is not None:
            model_limit = int(self.max_prompt_length) + int(self.max_completion_length)

        signature = (
            step_token,
            stop_token,
            max_steps,
            tokens_per_step,
            conf_scoring,
            conf_group,
            conf_aggregation,
            return_all,
            return_eos,
            random_sampling,
            smc_topk,
            conf_window,
            conf_eta,
            cdf_alpha,
            include_stop,
            self.num_generations,
            self.pad_token_id,
            self.temperature,
            self.top_p,
            -1 if self.top_k is None else self.top_k,
            0.0 if self.min_p is None else self.min_p,
            self.repetition_penalty,
            int(self.max_completion_length),
            None if model_limit is None else int(model_limit),
            id(self.llm),
            prm_signature,
            log_wandb,
        )

        if getattr(self, "_smc_vllm_signature", None) != signature:
            sg = StepGeneration(
                max_steps=max_steps,
                step_token=step_token,
                tokens_per_step=tokens_per_step,
                stop_token=stop_token,
                include_stop_str_in_output=include_stop,
            )

            self._smc_vllm = SMCVLLM(
                llm=self.llm,
                tokenizer=self.processing_class,
                num_particles=self.num_generations,
                pad_token_id=self.pad_token_id,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=-1 if self.top_k is None else self.top_k,
                min_p=0.0 if self.min_p is None else self.min_p,
                repetition_penalty=self.repetition_penalty,
                sg=sg,
                smc_topk=smc_topk,
                window_size=conf_window,
                max_new_tokens=int(self.max_completion_length),
                max_model_len=model_limit,
                scoring=conf_scoring,
                confidence_group=conf_group,
                confidence_aggregation=conf_aggregation,
                confidence_eta=conf_eta,
                cdf_alpha=cdf_alpha,
                return_all=return_all,
                return_eos=return_eos,
                wandb_logging=log_wandb,
                prm=prm_model,
                random_sampling=random_sampling,
                confidence_from_base=conf_from_base,
            )
            self._smc_vllm_signature = signature

        return self._smc_vllm

    def _generate_and_score_completions(
        self, inputs: list[dict[str, Union[torch.Tensor, Any]]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"

        prompts = [x["prompt"] for x in inputs]

        # We don't yet support visual reward models/function, so we keep a copy of the original text-only prompts for
        # later use in the reward computation. If images are present, we insert {"type": "image"} as required by the
        # VLM chat template.
        original_prompts = copy.deepcopy(prompts)

        # If the prompts are conversational and the inputs contain images, we need to convert the prompts from
        # [{"role": "user", "content": "What color is the sky?"}] to
        # [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "What color is the sky?"}]}]
        kwargs = {}
        images = None
        has_images = "image" in inputs[0]
        if has_images:
            images = [example.get("image") for example in inputs]
            kwargs = {"images": [[img] for img in images]}
            for prompt in prompts:
                if isinstance(prompt, list):
                    for message in prompt:
                        if not isinstance(message, dict):
                            continue
                        content = message.get("content")
                        role = message.get("role")
                        if isinstance(content, str):
                            if role == "user":
                                message["content"] = [{"type": "image"}, {"type": "text", "text": content}]
                            elif role == "system":
                                message["content"] = [{"type": "text", "text": content}]

        prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs]

        if self.max_prompt_length is not None:
            trunc_inputs = self.processing_class(
                text=prompts_text,
                return_tensors="pt",
                padding=True,
                padding_side="left",
                add_special_tokens=False,
                **kwargs,
            )
            trunc_inputs = super()._prepare_inputs(trunc_inputs)
            trunc_ids, trunc_mask = trunc_inputs["input_ids"], trunc_inputs["attention_mask"]

            protected = [self.image_token_id, self.vision_start_token_id, self.vision_end_token_id]
            protected = [token for token in protected if token is not None]
            trunc_ids, trunc_mask = truncate_with_protected_tokens(
                trunc_ids,
                trunc_mask,
                self.max_prompt_length,
                protected,
            )

            truncated_prompts_text = self.processing_class.batch_decode(
                trunc_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            truncated_prompts_text = [
                re.sub(rf"^({re.escape(self.pad_token)})+", "", text) for text in truncated_prompts_text
            ]

            if self.image_token is not None:
                truncated_prompts_text = [
                    re.sub(rf"({re.escape(self.image_token)})+", self.image_token, text)
                    for text in truncated_prompts_text
                ]

            prompts_text = truncated_prompts_text

        smc_meta: Optional[_SMCBatchMeta] = None

        if (self.use_vllm or hasattr(self, "llm")):
            if self.state.global_step != self._last_loaded_step:
                self._move_model_to_vllm()
                self._last_loaded_step = self.state.global_step

            smc_runner = self._ensure_smc_vllm()
            if hasattr(smc_runner, "_call_index"):
                smc_runner._call_index = max(int(self.state.global_step) - 1, 0)
            lora_request = self.model.load_lora("grpo_trainer_lora_model", load_tensors=True)
            payload = smc_runner.generate(prompts_text, list(prompts_text), lora_request=lora_request)

            if _smc_is_bundle(payload):
                completion_ids_list, smc_meta = _smc_flatten_bundle(
                    payload,
                    int(self.max_completion_length),
                    self._smc_return_all_limit,
                )
                inputs, prompts, original_prompts, prompts_text, images = _smc_expand_batch(
                    inputs,
                    prompts,
                    original_prompts,
                    prompts_text,
                    images if has_images else None,
                    self.num_generations,
                    smc_meta,
                )
                has_images = images is not None
                kwargs = {"images": [[img] for img in images]} if has_images else {}
            else:
                completion_ids_list = payload
                smc_meta = _smc_build_uniform_meta(
                    len(completion_ids_list), self.num_generations, int(self.max_completion_length)
                )

            prompt_inputs = self.processing_class(
                text=prompts_text,
                return_tensors="pt",
                padding=True,
                padding_side="left",
                add_special_tokens=False,
                **kwargs,
            )
            prompt_inputs = super()._prepare_inputs(prompt_inputs)
            prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]

            if self.max_prompt_length is not None:
                protected = [self.image_token_id, self.vision_start_token_id, self.vision_end_token_id]
                protected = [token for token in protected if token is not None]
                prompt_ids, prompt_mask = truncate_with_protected_tokens(
                    prompt_ids, prompt_mask, self.max_prompt_length, protected
                )

                prompts_text = self.processing_class.batch_decode(
                    prompt_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
                )
                prompts_text = [re.sub(rf"^({re.escape(self.pad_token)})+", "", text) for text in prompts_text]

                if self.image_token is not None:
                    prompts_text = [
                        re.sub(rf"({re.escape(self.image_token)})+", self.image_token, text) for text in prompts_text
                    ]

            raw_lengths = torch.tensor(
                [len(ids) for ids in completion_ids_list],
                device=prompt_ids.device,
                dtype=torch.long,
            )
            completion_ids = [torch.tensor(ids, device=prompt_ids.device) for ids in completion_ids_list]
            completion_ids = pad(completion_ids, padding_value=self.pad_token_id, padding_side="right")
            prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)

        else:
            # Default path: route through HF custom_generate (your existing SMC/HF path)
            # logging in generate function
            logging_config = {
                "is_main_process": self.is_world_process_zero(),
                "report_to": self.args.report_to,
                "global_step": self.state.global_step,
            }

            prompt_inputs = self.processing_class(
                text=prompts_text,
                return_tensors="pt",
                padding=True,
                padding_side="left",
                add_special_tokens=False,
                **kwargs,
            )
            prompt_inputs = super()._prepare_inputs(prompt_inputs)
            prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]

            if self.max_prompt_length is not None:
                protected = [self.image_token_id, self.vision_start_token_id, self.vision_end_token_id]
                protected = [token for token in protected if token is not None]
                prompt_ids, prompt_mask = truncate_with_protected_tokens(
                    prompt_ids, prompt_mask, self.max_prompt_length, protected
                )

                prompts_text = self.processing_class.batch_decode(
                    prompt_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
                )
                prompts_text = [re.sub(rf"^({re.escape(self.pad_token)})+", "", text) for text in prompts_text]

                if self.image_token is not None:
                    prompts_text = [
                        re.sub(rf"({re.escape(self.image_token)})+", self.image_token, text) for text in prompts_text
                    ]

            with (
                profiling_context(self, "transformers.generate"),
                unwrap_model_for_generation(
                    self.model_wrapped, self.accelerator, gather_deepspeed3_params=self.args.ds3_gather_for_generation
                ) as unwrapped_model,
                torch.no_grad(),
                FSDP.summon_full_params(self.model_wrapped, recurse=False) if self.is_fsdp_enabled else nullcontext(),
            ):
                prompt_inputs["input_ids"], prompt_inputs["attention_mask"] = prompt_ids, prompt_mask
                prompt_completion_ids = unwrapped_model.generate(
                    **prompt_inputs,
                    custom_generate='.',
                    generation_config=self.generation_config,
                    disable_compile=True,
                    logging_config=logging_config,
            )
            # Compute prompt length and extract completion ids
            prompt_length = prompt_ids.size(1)
            prompt_ids = prompt_completion_ids[:, :prompt_length]
            completion_ids = prompt_completion_ids[:, prompt_length:]
            smc_meta = _smc_build_uniform_meta(
                completion_ids.size(0), self.num_generations, int(self.max_completion_length)
            )

        if smc_meta is None:
            num_samples = completion_ids.size(0) if isinstance(completion_ids, torch.Tensor) else len(completion_ids)
            smc_meta = _smc_build_uniform_meta(
                num_samples,
                self.num_generations,
                int(self.max_completion_length),
            )

        # Mask everything after the first EOS token (note: vLLM strips the EOS token itself)
        is_eos = completion_ids == self.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        # Convert tensor to a list of lists of token IDs. This will be passed to the reward function, avoiding the need
        # to re-tokenize completions if the reward is computed from tokens.
        if raw_lengths is not None:
            completion_ids_list = [
                [int(token) for token in seq[:length.item()]]
                for seq, length in zip(completion_ids_list, raw_lengths)
            ]
        else:
            completion_ids_list = [
                [id.item() for id, m in zip(row, mask_row) if m]
                for row, mask_row in zip(completion_ids, completion_mask)
            ]

        # Sequence lengths for logging/truncation
        if raw_lengths is not None:
            lengths_for_mask = raw_lengths
        else:
            lengths_for_mask = completion_mask.sum(1)
        completion_lengths = lengths_for_mask.clone()

        max_length_cap = int(self.max_completion_length)

        # If mask_truncated_completions is enabled, zero out truncated completions in completion_mask
        if self.mask_truncated_completions:
            truncated_completions = lengths_for_mask >= max_length_cap
            completion_mask = completion_mask * (~truncated_completions).unsqueeze(1).int()

        # Determine the model's maximum supported sequence length (prompt + completion).
        model_max_length = (
            getattr(self.model.config, "max_position_embeddings", None)
            or getattr(self.model.config, "max_sequence_length", None)
            or getattr(self.processing_class, "model_max_length", None)
            or getattr(self.processing_class, "max_length", None)
        )
        if model_max_length is not None:
            model_max_length = int(model_max_length)
            max_prompt_allowed = max(model_max_length - 1, 1)
            if prompt_ids.size(1) > max_prompt_allowed:
                trim_prompt = prompt_ids.size(1) - max_prompt_allowed
                prompt_ids = prompt_ids[:, trim_prompt:].contiguous()
                prompt_mask = prompt_mask[:, trim_prompt:].contiguous()
                if "input_ids" in prompt_inputs:
                    prompt_inputs["input_ids"] = prompt_ids
                    prompt_inputs["attention_mask"] = prompt_mask

        # Align completion tensors to the maximum generated length and respect model constraints to avoid gather
        # mismatches downstream.
        effective_keep = int(completion_mask.sum(dim=1).max().item())
        effective_keep = max(effective_keep, 1)
        if model_max_length is not None:
            max_completion_allowed = max(model_max_length - prompt_ids.size(1), 1)
            effective_keep = min(effective_keep, max_completion_allowed)

        if self.max_completion_length is not None:
            # Guard against generation overshooting the configured cap; keep tensors mutually aligned.
            effective_keep = min(effective_keep, max(int(self.max_completion_length), 1))

        if completion_ids.size(1) != effective_keep:
            completion_ids = completion_ids[:, :effective_keep].contiguous()
        if completion_mask.size(1) != effective_keep:
            completion_mask = completion_mask[:, :effective_keep].contiguous()

        if raw_lengths is not None:
            raw_lengths = raw_lengths.clamp_max(effective_keep)

        completion_lengths = completion_lengths.clamp_max(effective_keep)

        # Rebuild concatenated ids/masks after potential trimming so shapes stay consistent.
        prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # (B, P+C)

        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens
        batch_size = self.args.per_device_train_batch_size if mode == "train" else self.args.per_device_eval_batch_size

        def _enforce_keep_len(new_keep: int) -> None:
            nonlocal logits_to_keep, completion_ids, completion_mask, completion_lengths, completion_ids_list
            nonlocal prompt_completion_ids, attention_mask, effective_keep, raw_lengths
            new_keep = int(new_keep)
            if new_keep <= 0 or new_keep >= logits_to_keep:
                return

            logits_to_keep = new_keep
            effective_keep = new_keep
            completion_ids = completion_ids[:, :new_keep].contiguous()
            completion_mask = completion_mask[:, :new_keep].contiguous()

            if raw_lengths is not None:
                raw_lengths = raw_lengths.clamp_max(new_keep)

            completion_lengths = completion_lengths.clamp_max(new_keep)
            completion_ids_list = [seq[:new_keep] for seq in completion_ids_list]

            prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
            attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)

        with torch.no_grad():
            # If the generation and optimization steps are misaligned—i.e., if generation does not occur at the end of
            # a full optimizer step (when gradient_accumulation_steps is not a multiple of generate_every)—then the
            # samples may come from an earlier version of the model. In that case, we need to track old_per_token_logps
            # for importance sampling. If the steps are aligned, importance sampling isn't necessary and we set
            # old_per_token_logps to None.
            generate_every = self.args.steps_per_generation * self.num_iterations  # generation frequency
            if self.args.gradient_accumulation_steps % generate_every != 0:
                old_per_token_logps, _, keep_len = self._get_per_token_logps_and_entropies(
                    self.model,
                    prompt_completion_ids,
                    attention_mask,
                    logits_to_keep,
                    batch_size,
                    pixel_values=prompt_inputs.get("pixel_values"),
                    image_grid_thw=prompt_inputs.get("image_grid_thw"),
                    pixel_attention_mask=prompt_inputs.get("pixel_attention_mask"),
                    image_sizes=prompt_inputs.get("image_sizes"),
                )
                _enforce_keep_len(keep_len)
                if old_per_token_logps.size(1) > logits_to_keep:
                    old_per_token_logps = old_per_token_logps[:, -logits_to_keep:]
            else:
                old_per_token_logps = None

            # Compute the per-token log probabilities for the reference model
            if self.beta != 0.0:
                if self.ref_model is not None:
                    ref_per_token_logps, _, keep_len = self._get_per_token_logps_and_entropies(
                        self.ref_model,
                        prompt_completion_ids,
                        attention_mask,
                        logits_to_keep,
                        batch_size=batch_size,
                        pixel_values=prompt_inputs.get("pixel_values"),
                        image_grid_thw=prompt_inputs.get("image_grid_thw"),
                        pixel_attention_mask=prompt_inputs.get("pixel_attention_mask"),
                        image_sizes=prompt_inputs.get("image_sizes"),
                    )
                    _enforce_keep_len(keep_len)
                    if old_per_token_logps is not None and old_per_token_logps.size(1) > logits_to_keep:
                        old_per_token_logps = old_per_token_logps[:, -logits_to_keep:]
                    if ref_per_token_logps.size(1) > logits_to_keep:
                        ref_per_token_logps = ref_per_token_logps[:, -logits_to_keep:]
                else:
                    with self.accelerator.unwrap_model(self.model).disable_adapter():
                        ref_per_token_logps, _, keep_len = self._get_per_token_logps_and_entropies(
                            self.model,
                            prompt_completion_ids,
                            attention_mask,
                            logits_to_keep,
                            batch_size=batch_size,
                            pixel_values=prompt_inputs.get("pixel_values"),
                            image_grid_thw=prompt_inputs.get("image_grid_thw"),
                            pixel_attention_mask=prompt_inputs.get("pixel_attention_mask"),
                            image_sizes=prompt_inputs.get("image_sizes"),
                        )
                        _enforce_keep_len(keep_len)
                        if old_per_token_logps is not None and old_per_token_logps.size(1) > logits_to_keep:
                            old_per_token_logps = old_per_token_logps[:, -logits_to_keep:]
                        if ref_per_token_logps.size(1) > logits_to_keep:
                            ref_per_token_logps = ref_per_token_logps[:, -logits_to_keep:]
            else:
                ref_per_token_logps = None

        # Decode the generated completions
        completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        if is_conversational(inputs[0]):
            completions = []
            for prompt, completion in zip(prompts, completions_text):
                bootstrap = prompt.pop()["content"] if prompt[-1]["role"] == "assistant" else ""
                completions.append([{"role": "assistant", "content": bootstrap + completion}])
        else:
            completions = completions_text
        # Calculate rewards for each reward function. rewards_per_func aggregates rewards across all processes. This is
        # important because rewards will be normalized per group, and completions are distributed. We will later slice
        # rewards_per_func to extract each process's subset.
        rewards_per_func = self._calculate_rewards(inputs, original_prompts, completions, completion_ids_list)

        if smc_meta is None:
            raise RuntimeError("SMC metadata is undefined after completion generation")

        local_group_counts = smc_meta.group_counts
        local_group_indices = smc_meta.group_indices
        local_drop_steps = smc_meta.drop_steps
        local_total_steps = smc_meta.total_steps
        local_kind_values = [kind.value for kind in smc_meta.kinds]
        local_conf_values = smc_meta.confidences

        def _ensure_list(obj):
            return obj if isinstance(obj, list) else [obj]

        def _ensure_nested_list(obj):
            if isinstance(obj, list) and (not obj or isinstance(obj[0], list)):
                return obj
            return [obj]

        group_segments = _ensure_list(gather_object(len(local_group_counts)))
        group_offsets: list[int] = []
        total_groups = 0
        for seg_size in group_segments:
            group_offsets.append(total_groups)
            total_groups += seg_size
        num_groups_global = total_groups

        group_offset = group_offsets[self.accelerator.process_index]
        local_group_indices_global = [group_offset + idx for idx in local_group_indices]

        group_indices_across = _ensure_nested_list(gather_object(local_group_indices_global))
        kinds_across = _ensure_nested_list(gather_object(local_kind_values))
        drop_steps_across = _ensure_nested_list(gather_object(local_drop_steps))
        total_steps_across = _ensure_nested_list(gather_object(local_total_steps))
        confidences_across = _ensure_nested_list(gather_object(local_conf_values))
        sample_counts_across = _ensure_list(gather_object(len(local_group_indices_global)))

        global_group_indices = [idx for chunk in group_indices_across for idx in chunk]
        global_kind_values = [kind for chunk in kinds_across for kind in chunk]
        global_drop_steps = [step for chunk in drop_steps_across for step in chunk]
        global_total_steps = [step for chunk in total_steps_across for step in chunk]
        global_confidences = [conf for chunk in confidences_across for conf in chunk]

        if rewards_per_func.size(0) != len(global_group_indices):
            raise ValueError("SMC metadata misalignment with reward tensor")
        if len(global_confidences) != len(global_group_indices):
            raise ValueError("SMC confidence metadata misalignment with reward tensor")
        if self._enable_drop_bonus:
            drop_bonus = _smc_drop_bonus(
                [_SMCTrajectoryKind(value) for value in global_kind_values],
                global_drop_steps,
                global_total_steps,
                self._smc_drop_reward_scheme,
                device,
            )
            rewards_per_func = torch.cat(
                [rewards_per_func, drop_bonus.unsqueeze(1)],
                dim=1,
            )
        
        reward_weights = self.reward_weights.to(device).unsqueeze(0)
        weighted_rewards = (rewards_per_func * reward_weights).nansum(dim=1)
        confidences_tensor = torch.tensor(global_confidences, dtype=torch.float32, device=device)
        weighted_with_conf = weighted_rewards * confidences_tensor

        num_legacy_global = sum(1 for kind in global_kind_values if kind == _SMCTrajectoryKind.LEGACY.value)
        downsample_enabled = (
            not self._downsample_disabled
            and self._num_generations_grad_effective > 0
            and num_legacy_global > self._num_generations_grad_effective
        )

        keep_mask_list: list[bool]
        if downsample_enabled:
            keep_mask_list = _build_legacy_keep_mask(
                global_group_indices,
                global_kind_values,
                weighted_with_conf,
                self._num_generations_grad_effective,
            )
        else:
            keep_mask_list = [True] * len(global_group_indices)

        legacy_kept = sum(
            1
            for keep, kind in zip(keep_mask_list, global_kind_values)
            if keep and kind == _SMCTrajectoryKind.LEGACY.value
        )
        legacy_dropped = num_legacy_global - legacy_kept

        keep_mask = torch.tensor(keep_mask_list, dtype=torch.bool, device=weighted_rewards.device)

        rewards_per_func = rewards_per_func[keep_mask]
        weighted_rewards = weighted_rewards[keep_mask]
        weighted_with_conf = weighted_with_conf[keep_mask]
        confidences_tensor = confidences_tensor[keep_mask]
        global_group_indices = [idx for idx, keep in zip(global_group_indices, keep_mask_list) if keep]
        global_kind_values = [kind for kind, keep in zip(global_kind_values, keep_mask_list) if keep]
        global_drop_steps = [step for step, keep in zip(global_drop_steps, keep_mask_list) if keep]
        global_total_steps = [step for step, keep in zip(global_total_steps, keep_mask_list) if keep]
        global_confidences = [conf for conf, keep in zip(global_confidences, keep_mask_list) if keep]

        orig_sample_counts = sample_counts_across
        orig_offsets: list[int] = []
        running = 0
        for count in orig_sample_counts:
            orig_offsets.append(running)
            running += count

        keep_counts_across: list[int] = []
        for count, offset in zip(orig_sample_counts, orig_offsets):
            segment_mask = keep_mask[offset : offset + count]
            keep_counts_across.append(int(segment_mask.sum().item()))

        local_orig_offset = orig_offsets[self.accelerator.process_index]
        local_orig_count = orig_sample_counts[self.accelerator.process_index]
        local_keep_mask_tensor = keep_mask[local_orig_offset : local_orig_offset + local_orig_count]
        local_keep_mask_list = local_keep_mask_tensor.cpu().tolist()

        sample_counts_across = keep_counts_across

        local_keep_mask_tensor = local_keep_mask_tensor.to(prompt_ids.device)

        prompt_ids = prompt_ids[local_keep_mask_tensor]
        prompt_mask = prompt_mask[local_keep_mask_tensor]
        completion_ids = completion_ids[local_keep_mask_tensor]
        completion_mask = completion_mask[local_keep_mask_tensor]
        if old_per_token_logps is not None:
            old_per_token_logps = old_per_token_logps[local_keep_mask_tensor]
        if ref_per_token_logps is not None:
            ref_per_token_logps = ref_per_token_logps[local_keep_mask_tensor]

        completion_ids_list = [
            seq for seq, keep in zip(completion_ids_list, local_keep_mask_list) if keep
        ]
        completions_text = [
            text for text, keep in zip(completions_text, local_keep_mask_list) if keep
        ]
        prompts_text = [
            text for text, keep in zip(prompts_text, local_keep_mask_list) if keep
        ]
        prompts = [prompt for prompt, keep in zip(prompts, local_keep_mask_list) if keep]
        original_prompts = [
            prompt for prompt, keep in zip(original_prompts, local_keep_mask_list) if keep
        ]
        if has_images and images is not None:
            images = [img for img, keep in zip(images, local_keep_mask_list) if keep]
            has_images = len(images) > 0

        completion_lengths = torch.tensor(
            [len(seq) for seq in completion_ids_list],
            device=device,
            dtype=torch.long,
        )
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)

        filtered_group_indices = [
            idx for idx, keep in zip(local_group_indices, local_keep_mask_list) if keep
        ]
        filtered_drop_steps = [
            step for step, keep in zip(local_drop_steps, local_keep_mask_list) if keep
        ]
        filtered_total_steps = [
            step for step, keep in zip(local_total_steps, local_keep_mask_list) if keep
        ]
        filtered_kinds = [
            kind for kind, keep in zip(smc_meta.kinds, local_keep_mask_list) if keep
        ]
        filtered_confidences = [
            conf for conf, keep in zip(smc_meta.confidences, local_keep_mask_list) if keep
        ]
        new_group_counts = [0] * len(local_group_counts)
        for idx in filtered_group_indices:
            new_group_counts[idx] += 1
        smc_meta = _SMCBatchMeta(
            new_group_counts,
            filtered_group_indices,
            filtered_drop_steps,
            filtered_total_steps,
            filtered_kinds,
            filtered_confidences,
        )
        local_group_counts = smc_meta.group_counts
        local_group_indices = smc_meta.group_indices
        local_drop_steps = smc_meta.drop_steps
        local_total_steps = smc_meta.total_steps
        local_kind_values = [kind.value for kind in smc_meta.kinds]
        local_group_indices_global = [group_offset + idx for idx in local_group_indices]

        prompt_inputs["input_ids"] = prompt_ids
        prompt_inputs["attention_mask"] = prompt_mask
        for key in ("pixel_values", "image_grid_thw", "pixel_attention_mask", "image_sizes"):
            if key not in prompt_inputs:
                continue
            value = prompt_inputs[key]
            if isinstance(value, torch.Tensor):
                prompt_inputs[key] = value[local_keep_mask_tensor]
            elif isinstance(value, (list, tuple)):
                filtered_items = [item for item, keep in zip(value, local_keep_mask_list) if keep]
                prompt_inputs[key] = type(value)(filtered_items) if isinstance(value, tuple) else filtered_items

        training_rewards_tensor = weighted_with_conf if self._smc_self_reward else weighted_rewards

        if num_groups_global == 0:
            group_mean = torch.zeros(0, dtype=torch.float32, device=device)
            group_std = torch.zeros(0, dtype=torch.float32, device=device)
            std_zero_mask = torch.zeros(0, dtype=torch.bool, device=device)
            advantages_all = training_rewards_tensor
            base_group_mean = group_mean
            base_group_std = group_std
            self_group_mean = group_mean
            self_group_std = group_std
        else:
            group_indices_tensor = torch.tensor(global_group_indices, dtype=torch.long, device=device)
            group_sum = torch.bincount(
                group_indices_tensor, weights=training_rewards_tensor, minlength=num_groups_global
            )
            group_sq_sum = torch.bincount(
                group_indices_tensor, weights=training_rewards_tensor**2, minlength=num_groups_global
            )
            group_count = torch.bincount(group_indices_tensor, minlength=num_groups_global).clamp(min=1)
            group_mean = group_sum / group_count
            group_var = torch.clamp(group_sq_sum / group_count - group_mean**2, min=0.0)
            group_std = torch.sqrt(group_var)
            std_zero_mask = group_std == 0
            sample_mean = group_mean[group_indices_tensor]
            sample_std = group_std[group_indices_tensor]
            advantages_all = training_rewards_tensor - sample_mean
            if self.scale_rewards:
                advantages_all = advantages_all / (sample_std + 1e-4)

            base_group_sum = torch.bincount(
                group_indices_tensor, weights=weighted_rewards, minlength=num_groups_global
            )
            base_group_sq_sum = torch.bincount(
                group_indices_tensor, weights=weighted_rewards**2, minlength=num_groups_global
            )
            base_group_mean = base_group_sum / group_count
            base_group_var = torch.clamp(base_group_sq_sum / group_count - base_group_mean**2, min=0.0)
            base_group_std = torch.sqrt(base_group_var)
            self_group_sum = torch.bincount(
                group_indices_tensor, weights=weighted_with_conf, minlength=num_groups_global
            )
            self_group_sq_sum = torch.bincount(
                group_indices_tensor, weights=weighted_with_conf**2, minlength=num_groups_global
            )
            self_group_mean = self_group_sum / group_count
            self_group_var = torch.clamp(self_group_sq_sum / group_count - self_group_mean**2, min=0.0)
            self_group_std = torch.sqrt(self_group_var)

        sample_offsets: list[int] = []
        running = 0
        for count in sample_counts_across:
            sample_offsets.append(running)
            running += count
        local_offset = sample_offsets[self.accelerator.process_index]
        local_count = len(local_group_indices_global)
        advantages = advantages_all[local_offset : local_offset + local_count].clone()
        all_process_advantages = advantages_all.clone()
        local_confidences_tensor = confidences_tensor[local_offset : local_offset + local_count]
        local_weighted_with_conf = weighted_with_conf[local_offset : local_offset + local_count]

        mean_grouped_rewards = group_mean
        std_grouped_rewards = group_std
        is_std_zero = std_zero_mask

        # Log the metrics
        if mode == "train":
            self.state.num_input_tokens_seen += self.accelerator.gather(attention_mask.sum()).sum().item()
        self._metrics[mode]["num_tokens"] = [self.state.num_input_tokens_seen]

        max_length_cap = int(self.max_completion_length)

        # Log completion lengths, mean, min, max
        agg_completion_lengths = self.accelerator.gather(completion_lengths)
        self._metrics[mode]["completions/mean_length"].append(agg_completion_lengths.float().mean().item())
        self._metrics[mode]["completions/min_length"].append(agg_completion_lengths.float().min().item())
        self._metrics[mode]["completions/max_length"].append(agg_completion_lengths.float().max().item())

        # Identify sequences that terminated with EOS and log their lengths
        terminated_flags = (completion_lengths < max_length_cap).int()
        agg_terminated_with_eos = self.accelerator.gather(terminated_flags).bool()
        term_completion_lengths = agg_completion_lengths[agg_terminated_with_eos]
        clipped_completions_ratio = 1 - len(term_completion_lengths) / len(agg_completion_lengths)
        self._metrics[mode]["completions/clipped_ratio"].append(clipped_completions_ratio)
        if len(term_completion_lengths) == 0:  # edge case where no terminated sequences are found
            term_completion_lengths = torch.zeros(1, device=device)
        self._metrics[mode]["completions/mean_terminated_length"].append(term_completion_lengths.float().mean().item())
        self._metrics[mode]["completions/min_terminated_length"].append(term_completion_lengths.float().min().item())
        self._metrics[mode]["completions/max_terminated_length"].append(term_completion_lengths.float().max().item())

        # Calculate mean reward per function, but only for samples where the function was applied (non-NaN values)
        for i, reward_func_name in enumerate(self.reward_func_names):
            mean_rewards = torch.nanmean(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}/mean"].append(mean_rewards)
            std_rewards = nanstd(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}/std"].append(std_rewards)
        self._metrics[mode]["reward"].append(
            mean_grouped_rewards.mean().item() if mean_grouped_rewards.numel() > 0 else 0.0
        )
        self._metrics[mode]["reward_std"].append(
            std_grouped_rewards.mean().item() if std_grouped_rewards.numel() > 0 else 0.0
        )
        self._metrics[mode]["frac_reward_zero_std"].append(is_std_zero.float().mean().item())
        self._metrics[mode]["reward_original/mean"].append(
            base_group_mean.mean().item() if base_group_mean.numel() > 0 else 0.0
        )
        self._metrics[mode]["reward_original/std"].append(
            base_group_std.mean().item() if base_group_std.numel() > 0 else 0.0
        )
        self._metrics[mode]["reward_self/mean"].append(
            self_group_mean.mean().item() if self_group_mean.numel() > 0 else 0.0
        )
        self._metrics[mode]["reward_self/std"].append(
            self_group_std.mean().item() if self_group_std.numel() > 0 else 0.0
        )
        agg_confidences = self.accelerator.gather(confidences_tensor)
        if agg_confidences.numel() > 0:
            agg_confidences = agg_confidences.float()
            self._metrics[mode]["self_confidence/mean"].append(agg_confidences.mean().item())
            self._metrics[mode]["self_confidence/std"].append(agg_confidences.std(unbiased=False).item())
        agg_self_rewards = self.accelerator.gather(weighted_with_conf)
        if agg_self_rewards.numel() > 0:
            agg_self_rewards = agg_self_rewards.float()
            self._metrics[mode]["self_reward/mean"].append(agg_self_rewards.mean().item())
            self._metrics[mode]["self_reward/std"].append(agg_self_rewards.std(unbiased=False).item())
        self._metrics[mode]["downsample/legacy_dropped"].append(float(max(legacy_dropped, 0)))

        # Log prompt and completion texts
        self._logs["prompt"].extend(gather_object(prompts_text))
        self._logs["completion"].extend(gather_object(completions_text))
        for i, name in enumerate(self.reward_func_names):
            self._logs["rewards"][name].extend(rewards_per_func[:, i].tolist())
        self._logs["advantages"].extend(all_process_advantages.tolist())
        self._logs["self_confidence"].extend(confidences_tensor.detach().cpu().tolist())
        self._logs["self_weighted_reward"].extend(weighted_with_conf.detach().cpu().tolist())

        if has_images:
            self._logs["image"].extend(gather_object(images))

        output = {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "advantages": advantages,
        }
        if old_per_token_logps is not None:
            output["old_per_token_logps"] = old_per_token_logps
        if ref_per_token_logps is not None:
            output["ref_per_token_logps"] = ref_per_token_logps
        if "pixel_values" in prompt_inputs:
            output["pixel_values"] = prompt_inputs["pixel_values"]
        if "image_grid_thw" in prompt_inputs:
            output["image_grid_thw"] = prompt_inputs["image_grid_thw"]
        if "pixel_attention_mask" in prompt_inputs:
            output["pixel_attention_mask"] = prompt_inputs["pixel_attention_mask"]
        if "image_sizes" in prompt_inputs:
            output["image_sizes"] = prompt_inputs["image_sizes"]
        return output

    def compute_liger_loss(self, unwrapped_model, inputs):
        # Compute the per-token log probabilities for the model
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens

        # Get the last hidden state of the model
        last_hidden_state = self._get_last_hidden_state(
            unwrapped_model,
            input_ids,
            attention_mask,
            logits_to_keep,
            inputs.get("pixel_values"),
            inputs.get("image_grid_thw"),
            inputs.get("pixel_attention_mask"),
            inputs.get("image_sizes"),
        )

        # compute loss and metrics using liger grpo loss
        loss, metrics = self.liger_grpo_loss(
            _input=last_hidden_state,
            lin_weight=unwrapped_model.lm_head.weight,
            selected_token_ids=completion_ids,
            attention_mask=completion_mask,
            advantages=inputs["advantages"],
            bias=unwrapped_model.lm_head.bias,
            old_per_token_logps=inputs.get("old_per_token_logps"),
            ref_per_token_logps=inputs.get("ref_per_token_logps"),
        )
        # Extract metrics from the liger_grpo_loss output
        # KL divergence is the first metric when beta is non-zero
        mean_kl = metrics[0] if self.beta != 0.0 else None
        clip_ratio = metrics[-1]

        mode = "train" if self.model.training else "eval"
        if self.beta != 0.0:
            self._metrics[mode]["kl"].append(self.accelerator.gather(mean_kl).mean().item())
        self._metrics[mode]["clip_ratio"].append(self.accelerator.gather(clip_ratio).mean().item())
        return loss

    @profiling_decorator
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")
        if self.use_liger_loss:
            # Compute the loss using the liger grpo loss
            unwrapped_model = self.accelerator.unwrap_model(model)
            return self._forward_redirection(model, unwrapped_model, self.compute_liger_loss, unwrapped_model, inputs)
        else:
            return self._compute_loss(model, inputs)

    def _compute_loss(self, model, inputs):
        # Compute the per-token log probabilities for the model
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens

        # Compute the per_token_logps and the entropy at each position in the completion
        per_token_logps, entropies, _ = self._get_per_token_logps_and_entropies(
            model,
            input_ids,
            attention_mask,
            logits_to_keep,
            compute_entropy=True,
            pixel_values=inputs.get("pixel_values"),
            image_grid_thw=inputs.get("image_grid_thw"),
            pixel_attention_mask=inputs.get("pixel_attention_mask"),
            image_sizes=inputs.get("image_sizes"),
        )

        if self.top_entropy_quantile < 1.0:
            entropy_mask = self.get_high_entropy_mask(entropies, completion_mask, 1 - self.top_entropy_quantile)
        else:
            entropy_mask = None

        # Compute the KL divergence between the model and the reference model
        if self.beta != 0.0:
            ref_per_token_logps = inputs["ref_per_token_logps"]
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
            )

        # Compute the loss
        advantages = inputs["advantages"]
        # When using num_iterations == 1 and steps_per_generation <= gradient_accumulation_steps
        # old_per_token_logps == per_token_logps, so we can skip it's computation
        # (see _generate_and_score_completions) and use per_token_logps.detach() instead.
        old_per_token_logps = inputs.get("old_per_token_logps")
        old_per_token_logps = per_token_logps.detach() if old_per_token_logps is None else old_per_token_logps

        log_ratio = per_token_logps - old_per_token_logps
        if self.importance_sampling_level == "token":
            log_importance_weights = log_ratio
        elif self.importance_sampling_level == "sequence":
            log_importance_weights = (log_ratio * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)
            log_importance_weights = log_importance_weights.unsqueeze(-1)
        else:
            raise ValueError(
                f"Unknown importance sampling level: {self.importance_sampling_level}. Possible values are 'token' "
                "and 'sequence'."
            )
        # From here, log_importance_weights (and all subsequent tensors, coef_1, coef_2, etc.) shape depends on
        # importance_sampling_level: "token" level: (B, T); "sequence" level: (B, 1)

        coef_1 = torch.exp(log_importance_weights)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)

        # Two-sided clipping
        if self.args.delta is not None:
            coef_1 = torch.clamp(coef_1, max=self.args.delta)

        per_token_loss1 = coef_1 * advantages.unsqueeze(1)
        per_token_loss2 = coef_2 * advantages.unsqueeze(1)
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
        if entropy_mask is not None:
            per_token_loss = per_token_loss * entropy_mask
        if self.beta != 0.0:
            per_token_loss = per_token_loss + self.beta * per_token_kl

        if self.loss_type == "grpo":
            loss = ((per_token_loss * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)).mean()
        elif self.loss_type == "bnpo":
            loss = (per_token_loss * completion_mask).sum() / completion_mask.sum().clamp(min=1.0)
        elif self.loss_type == "dr_grpo":
            loss = (per_token_loss * completion_mask).sum() / (per_token_loss.size(0) * self.max_completion_length)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        # Log the metrics
        mode = "train" if self.model.training else "eval"

        completion_token_count = completion_mask.sum().clamp(min=1.0)

        def masked_batch_mean(x):
            if x.shape[1] == 1:  # when importance_sampling_level == "sequence"
                return x.mean()
            else:
                return (x * completion_mask).sum() / completion_token_count

        if self.beta != 0.0:
            mean_kl = masked_batch_mean(per_token_kl)
            self._metrics[mode]["kl"].append(self.accelerator.gather(mean_kl).nanmean().item())

        mean_entropy = masked_batch_mean(entropies)
        self._metrics[mode]["entropy"].append(self.accelerator.gather(mean_entropy).nanmean().item())

        # Compute the clipped probability ratios
        is_low_clipped = (coef_1 < 1 - self.epsilon_low) & (advantages.unsqueeze(1) < 0)
        is_high_clipped = (coef_1 > 1 + self.epsilon_high) & (advantages.unsqueeze(1) > 0)
        is_region_clipped = is_low_clipped | is_high_clipped

        low_clip = masked_batch_mean(is_low_clipped.float())
        high_clip = masked_batch_mean(is_high_clipped.float())
        clip_ratio = masked_batch_mean(is_region_clipped.float())

        gathered_low_clip = self.accelerator.gather(low_clip)
        self._metrics[mode]["clip_ratio/low_mean"].append(gathered_low_clip.nanmean().item())
        self._metrics[mode]["clip_ratio/low_min"].append(nanmin(gathered_low_clip).item())
        gathered_high_clip = self.accelerator.gather(high_clip)
        self._metrics[mode]["clip_ratio/high_mean"].append(gathered_high_clip.nanmean().item())
        self._metrics[mode]["clip_ratio/high_max"].append(nanmax(gathered_high_clip).item())
        gathered_clip_ratio = self.accelerator.gather(clip_ratio)
        self._metrics[mode]["clip_ratio/region_mean"].append(gathered_clip_ratio.nanmean().item())
        return loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys: Optional[list[str]] = None):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)
            loss = loss.mean().detach()
        return loss, None, None

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        mode = "train" if self.model.training else "eval"
        metrics = {key: sum(val) / len(val) for key, val in self._metrics[mode].items()}  # average the metrics

        # This method can be called both in training and evaluation. When called in evaluation, the keys in `logs`
        # start with "eval_". We need to add the prefix "eval_" to the keys in `metrics` to match the format.
        if mode == "eval":
            metrics = {f"eval_{key}": val for key, val in metrics.items()}

        logs = {**logs, **metrics}
        super().log(logs, start_time)
        self._metrics[mode].clear()

        if self.accelerator.is_main_process and self.log_completions:
            if is_rich_available():
                print_prompt_completions_sample(
                    self._logs["prompt"],
                    self._logs["completion"],
                    self._logs["rewards"],
                    self._logs["advantages"],
                    self.state.global_step,
                    self.num_completions_to_print,
                )

            if self.args.report_to and "wandb" in self.args.report_to and wandb.run is not None:
                import pandas as pd

                table = {
                    "step": [str(self.state.global_step)] * len(self._logs["prompt"]),
                    "prompt": self._logs["prompt"],
                    "completion": self._logs["completion"],
                    **self._logs["rewards"],
                    "advantage": self._logs["advantages"],
                    "self_confidence": self._logs["self_confidence"],
                    "self_weighted_reward": self._logs["self_weighted_reward"],
                }

                if self._logs["image"]:
                    table["image"] = []
                    for img in self._logs["image"]:
                        if img is not None:
                            # Convert images to wandb Image objects for proper visualization
                            table["image"].append(wandb.Image(img))
                        else:
                            table["image"].append(None)

                df = pd.DataFrame(table)
                if self.wandb_log_unique_prompts:
                    df = df.drop_duplicates(subset=["prompt"])
                wandb.log({"completions": wandb.Table(dataframe=df)})

    # Ensure the model card is saved along with the checkpoint
    def _save_checkpoint(self, model, trial):
        if self.args.hub_model_id is None:
            model_name = Path(self.args.output_dir).name
        else:
            model_name = self.args.hub_model_id.split("/")[-1]
        self.create_model_card(model_name=model_name)
        super()._save_checkpoint(model, trial)

    def create_model_card(
        self,
        model_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        tags: Union[str, list[str], None] = None,
    ):
        """
        Creates a draft of a model card using the information available to the `Trainer`.

        Args:
            model_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the model.
            dataset_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the dataset used for training.
            tags (`str`, `list[str]` or `None`, *optional*, defaults to `None`):
                Tags to be associated with the model card.
        """
        if not self.is_world_process_zero():
            return

        if hasattr(self.model.config, "_name_or_path") and not os.path.isdir(self.model.config._name_or_path):
            base_model = self.model.config._name_or_path
        else:
            base_model = None

        # normalize `tags` to a mutable set
        if tags is None:
            tags = set()
        elif isinstance(tags, str):
            tags = {tags}
        else:
            tags = set(tags)

        if hasattr(self.model.config, "unsloth_version"):
            tags.add("unsloth")

        tags.update(self._tag_names)

        citation = textwrap.dedent(
            """\
            @article{zhihong2024deepseekmath,
                title        = {{DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models}},
                author       = {Zhihong Shao and Peiyi Wang and Qihao Zhu and Runxin Xu and Junxiao Song and Mingchuan Zhang and Y. K. Li and Y. Wu and Daya Guo},
                year         = 2024,
                eprint       = {arXiv:2402.03300},
            }
            """
        )

        model_card = generate_model_card(
            base_model=base_model,
            model_name=model_name,
            hub_model_id=self.hub_model_id,
            dataset_name=dataset_name,
            tags=tags,
            wandb_url=wandb.run.url if is_wandb_available() and wandb.run is not None else None,
            comet_url=get_comet_experiment_url(),
            trainer_name="GRPO",
            trainer_citation=citation,
            paper_title="DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models",
            paper_id="2402.03300",
        )

        model_card.save(os.path.join(self.args.output_dir, "README.md"))
