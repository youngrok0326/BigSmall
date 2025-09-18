"""
Patch snippets for `trainer/smcgrpo_trainer.py`
Only the code below is intended for insertion/replacement.
"""

# ---------------------------------------------------------------------------
# 1) Add near the other imports
# ---------------------------------------------------------------------------
# from dataclasses import dataclass
# from enum import Enum

# ---------------------------------------------------------------------------
# 2) Place right after the RewardFunc alias
# ---------------------------------------------------------------------------
#
# class _SMCTrajectoryKind(str, Enum):
#     LEGACY = "legacy"
#     DROPPED = "dropped"
#
# @dataclass(slots=True)
# class _SMCBatchMeta:
#     group_counts: list[int]
#     group_indices: list[int]
#     drop_steps: list[int]
#     total_steps: list[int]
#     kinds: list[_SMCTrajectoryKind]
#
#     @property
#     def num_groups(self) -> int:
#         return len(self.group_counts)
#
#     @property
#     def num_samples(self) -> int:
#         return len(self.group_indices)
#
# def _smc_is_bundle(payload: Any) -> bool:
#     return isinstance(payload, dict) and "completions" in payload and "group_sizes" in payload
#
# def _smc_build_uniform_meta(num_samples: int, num_generations: int, fallback_steps: int) -> _SMCBatchMeta:
#     if num_generations <= 0:
#         raise ValueError("num_generations must be positive")
#     if num_samples % num_generations != 0:
#         raise ValueError(
#             f"Batch size {num_samples} must be divisible by num_generations {num_generations}."
#         )
#     num_groups = num_samples // num_generations
#     group_counts = [num_generations] * num_groups
#     group_indices = [g for g in range(num_groups) for _ in range(num_generations)]
#     drop_steps = [-1] * len(group_indices)
#     total_steps = [fallback_steps] * len(group_indices)
#     kinds = [_SMCTrajectoryKind.LEGACY] * len(group_indices)
#     return _SMCBatchMeta(group_counts, group_indices, drop_steps, total_steps, kinds)
#
# def _smc_flatten_bundle(payload: dict[str, Any], fallback_steps: int) -> tuple[list[list[int]], _SMCBatchMeta]:
#     completions_by_group: list[list[list[int]]] = payload.get("completions", [])  # type: ignore[assignment]
#     saved: list[list[tuple[int, list[int]]]] = payload.get("saved", [])  # type: ignore[assignment]
#     reasoning_steps: list[int] = payload.get("group_reasoning_steps", [])  # type: ignore[assignment]
#
#     flattened: list[list[int]] = []
#     group_counts: list[int] = []
#     group_indices: list[int] = []
#     drop_steps: list[int] = []
#     total_steps: list[int] = []
#     kinds: list[_SMCTrajectoryKind] = []
#
#     for group_idx, primary in enumerate(completions_by_group):
#         extras = saved[group_idx] if group_idx < len(saved) else []
#         step_cap = int(reasoning_steps[group_idx]) if group_idx < len(reasoning_steps) else fallback_steps
#
#         for seq in primary:
#             flattened.append(seq)
#             group_indices.append(group_idx)
#             drop_steps.append(-1)
#             total_steps.append(step_cap)
#             kinds.append(_SMCTrajectoryKind.LEGACY)
#
#         for step_idx, seq in extras:
#             flattened.append(seq)
#             group_indices.append(group_idx)
#             drop_steps.append(int(step_idx))
#             total_steps.append(max(step_cap, int(step_idx) + 1))
#             kinds.append(_SMCTrajectoryKind.DROPPED)
#
#         group_counts.append(len(primary) + len(extras))
#
#     meta = _SMCBatchMeta(group_counts, group_indices, drop_steps, total_steps, kinds)
#     return flattened, meta
#
# def _smc_expand_batch(
#     inputs: Sequence[dict[str, Any]],
#     prompts: Sequence[Any],
#     original_prompts: Sequence[Any],
#     prompts_text: Sequence[str],
#     images: Optional[Sequence[Any]],
#     num_generations: int,
#     meta: _SMCBatchMeta,
# ) -> tuple[list[dict[str, Any]], list[Any], list[Any], list[str], Optional[list[Any]]]:
#     if num_generations <= 0:
#         raise ValueError("num_generations must be positive")
#     if len(inputs) % num_generations != 0:
#         raise ValueError("SMC bundle size mismatch: batch is not a multiple of num_generations")
#
#     num_groups = len(inputs) // num_generations
#     base_inputs = [copy.deepcopy(inputs[g * num_generations]) for g in range(num_groups)]
#     base_prompts = [copy.deepcopy(prompts[g * num_generations]) for g in range(num_groups)]
#     base_original_prompts = [copy.deepcopy(original_prompts[g * num_generations]) for g in range(num_groups)]
#     base_prompts_text = [prompts_text[g * num_generations] for g in range(num_groups)]
#     base_images = None if images is None else [images[g * num_generations] for g in range(num_groups)]
#
#     expanded_inputs: list[dict[str, Any]] = []
#     expanded_prompts: list[Any] = []
#     expanded_original_prompts: list[Any] = []
#     expanded_prompts_text: list[str] = []
#     expanded_images: Optional[list[Any]] = [] if base_images is not None else None
#
#     for group_idx, count in enumerate(meta.group_counts):
#         for _ in range(count):
#             expanded_inputs.append(copy.deepcopy(base_inputs[group_idx]))
#             expanded_prompts.append(copy.deepcopy(base_prompts[group_idx]))
#             expanded_original_prompts.append(copy.deepcopy(base_original_prompts[group_idx]))
#             expanded_prompts_text.append(base_prompts_text[group_idx])
#             if expanded_images is not None and base_images is not None:
#                 expanded_images.append(base_images[group_idx])
#
#     return expanded_inputs, expanded_prompts, expanded_original_prompts, expanded_prompts_text, expanded_images
#
# def _smc_drop_bonus(
#     kinds: Sequence[_SMCTrajectoryKind],
#     drop_steps: Sequence[int],
#    total_steps: Sequence[int],
#     device: torch.device,
# ) -> torch.Tensor:
#     reward = torch.zeros(len(kinds), dtype=torch.float32, device=device)
#     if not kinds:
#         return reward
#     drop_mask = torch.tensor([kind == _SMCTrajectoryKind.DROPPED for kind in kinds], dtype=torch.bool, device=device)
#     if not drop_mask.any():
#         return reward
#     drop_tensor = torch.tensor(drop_steps, dtype=torch.float32, device=device)
#     total_tensor = torch.tensor(total_steps, dtype=torch.float32, device=device)
#     denom = torch.where(total_tensor <= 0, torch.ones_like(total_tensor), total_tensor)
#     reward[drop_mask] = 0.1 * torch.clamp(drop_tensor[drop_mask] + 1.0, min=0.0) / denom[drop_mask]
#     return reward

# ---------------------------------------------------------------------------
# 3) Replace block inside `_generate_and_score_completions`
#    starting from the comment “# SMC over colocated vLLM when enabled” up to the end of
#    the first `else:` branch (the one that handles vLLM output before falling back to HF).
# ---------------------------------------------------------------------------
#
#         smc_meta: _SMCBatchMeta | None = None
#
#         # SMC over colocated vLLM when enabled
#         if (self.use_vllm or hasattr(self, "llm")):
#             if self.state.global_step != self._last_loaded_step:
#                 self._move_model_to_vllm()
#                 self._last_loaded_step = self.state.global_step
#
#             smc_runner = self._ensure_smc_vllm()
#             payload = smc_runner.generate(prompts_text, original_prompts)
#
#             if _smc_is_bundle(payload):
#                 completion_ids_list, smc_meta = _smc_flatten_bundle(payload, int(self.max_completion_length))
#                 inputs, prompts, original_prompts, prompts_text, images = _smc_expand_batch(
#                     inputs,
#                     prompts,
#                     original_prompts,
#                     prompts_text,
#                     images,
#                     self.num_generations,
#                     smc_meta,
#                 )
#                 kwargs = {"images": [[img] for img in images]} if images is not None else {}
#             else:
#                 completion_ids_list = payload
#                 smc_meta = _smc_build_uniform_meta(len(completion_ids_list), self.num_generations, int(self.max_completion_length))
#
#             prompt_inputs = self.processing_class(
#                 text=prompts_text,
#                 return_tensors="pt",
#                 padding=True,
#                 padding_side="left",
#                 add_special_tokens=False,
#                 **kwargs,
#             )
#             prompt_inputs = super()._prepare_inputs(prompt_inputs)
#             prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]
#
#             if self.max_prompt_length is not None:
#                 protected = [self.image_token_id, self.vision_start_token_id, self.vision_end_token_id]
#                 protected = [token for token in protected if token is not None]
#                 prompt_ids, prompt_mask = truncate_with_protected_tokens(prompt_ids, prompt_mask, self.max_prompt_length, protected)
#
#                 prompts_text = self.processing_class.batch_decode(prompt_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
#                 prompts_text = [re.sub(rf"^({re.escape(self.pad_token)})+", "", text) for text in prompts_text]
#                 if self.image_token is not None:
#                     prompts_text = [re.sub(rf"({re.escape(self.image_token)})+", self.image_token, text) for text in prompts_text]
#
#             completion_ids = [torch.tensor(ids, device=prompt_ids.device) for ids in completion_ids_list]
#             completion_ids = pad(completion_ids, padding_value=self.pad_token_id, padding_side="right")
#             prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
#
#         else:
#             ...  # keep existing HF custom_generate branch, but set smc_meta = _smc_build_uniform_meta(...)

# ---------------------------------------------------------------------------
# 4) Replace the reward-normalisation block (downstream of `_calculate_rewards`)
# ---------------------------------------------------------------------------
# After `rewards_per_func = self._calculate_rewards(...)`, insert:
#
#         local_group_counts = smc_meta.group_counts if smc_meta is not None else [self.num_generations] * (len(rewards_per_func) // self.num_generations)
#         local_group_indices = smc_meta.group_indices if smc_meta is not None else [
#             idx for idx in range(len(local_group_counts)) for _ in range(local_group_counts[idx])
#         ]
#         local_drop_steps = smc_meta.drop_steps if smc_meta is not None else [-1] * len(local_group_indices)
#         local_total_steps = smc_meta.total_steps if smc_meta is not None else [int(self.max_completion_length)] * len(local_group_indices)
#         local_kinds = [kind.value for kind in (smc_meta.kinds if smc_meta is not None else [_SMCTrajectoryKind.LEGACY] * len(local_group_indices))]
#
#         group_counts_across = gather_object(local_group_counts)
#         group_offsets: list[int] = []
#         total = 0
#         for counts in group_counts_across:
#             group_offsets.append(total)
#             total += len(counts)
#         num_groups_global = total
#
#         group_offset = group_offsets[self.accelerator.process_index]
#         local_group_indices_global = [group_offset + idx for idx in local_group_indices]
#
#         group_indices_across = gather_object(local_group_indices_global)
#         kinds_across = gather_object(local_kinds)
#         drop_steps_across = gather_object(local_drop_steps)
#         total_steps_across = gather_object(local_total_steps)
#         sample_counts_across = gather_object(len(local_group_indices_global))
#
#         global_group_indices = [idx for chunk in group_indices_across for idx in chunk]
#         global_kinds = [kind for chunk in kinds_across for kind in chunk]
#         global_drop_steps = [step for chunk in drop_steps_across for step in chunk]
#         global_total_steps = [step for chunk in total_steps_across for step in chunk]
#
#         if rewards_per_func.size(0) != len(global_group_indices):
#             raise ValueError("SMC metadata misalignment with reward tensor")
#
#         drop_bonus = _smc_drop_bonus([_SMCTrajectoryKind(kind) for kind in global_kinds], global_drop_steps, global_total_steps, rewards_per_func.device)
#         if rewards_per_func.size(1) >= 1:
#             rewards_per_func[:, 0] = drop_bonus
#
#         reward_weights = self.reward_weights.to(rewards_per_func.device).unsqueeze(0)
#         weighted_rewards = (rewards_per_func * reward_weights).nansum(dim=1)
#
#         group_indices_tensor = torch.tensor(global_group_indices, dtype=torch.long, device=rewards_per_func.device)
#         ones = torch.ones_like(group_indices_tensor, dtype=torch.float32)
#         group_sum = torch.bincount(group_indices_tensor, weights=weighted_rewards, minlength=num_groups_global)
#         group_sq_sum = torch.bincount(group_indices_tensor, weights=weighted_rewards ** 2, minlength=num_groups_global)
#         group_count = torch.bincount(group_indices_tensor, minlength=num_groups_global).clamp(min=1)
#         group_mean = group_sum / group_count
#         group_std = torch.sqrt(torch.clamp(group_sq_sum / group_count - group_mean ** 2, min=0.0))
#
#         sample_mean = group_mean[group_indices_tensor]
#         sample_std = group_std[group_indices_tensor]
#         advantages_all = weighted_rewards - sample_mean
#         if self.scale_rewards:
#             advantages_all = advantages_all / (sample_std + 1e-4)
#
#         sample_offsets: list[int] = []
#         running = 0
#         for count in sample_counts_across:
#             sample_offsets.append(running)
#             running += count
#         local_offset = sample_offsets[self.accelerator.process_index]
#         local_count = len(local_group_indices_global)
#         advantages = advantages_all[local_offset : local_offset + local_count]
#         all_process_advantages = advantages_all.clone()
#
#         # Replace the old `view(-1, self.num_generations)` block, but keep the metric/logging
#         # updates below identical, except use `group_mean`, `group_std`, and `std_zero_mask = (group_std == 0)`.
#         std_zero_mask = group_std == 0
#         # (subsequent metric/logging code stays the same, using the new tensors)

# ---------------------------------------------------------------------------
# These snippets are intentionally minimal so performance matches the legacy path.
# Insert only the blocks you need, and remove the guiding comments when applying
# them to `trainer/smcgrpo_trainer.py`.
# ---------------------------------------------------------------------------
