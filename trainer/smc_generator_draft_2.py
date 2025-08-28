import torch
from torch import nn
import os
from typing import TYPE_CHECKING
from trl.trainer.utils import selective_log_softmax
import wandb

if TYPE_CHECKING:
    from .smcgrpo_trainer import TSMCGRPOTrainer


class SMCGenerator:
    def __init__(self, trainer: "TSMCGRPOTrainer"):
        # ... (Same as before)

    @torch.no_grad()
    def generate(self, unwrapped_model: nn.Module, prompt_ids: torch.Tensor, prompt_mask: torch.Tensor) -> dict:
        batch_size, prompt_len = prompt_ids.shape
        total_len = prompt_len + self.args.max_completion_length
        
        sequences = torch.full((batch_size, total_len), self.trainer.pad_token_id, dtype=torch.long, device=self.device)
        sequences[:, :prompt_len] = prompt_ids
        
        all_policy_logits = torch.empty(
            (batch_size, self.args.max_completion_length, unwrapped_model.config.vocab_size),
            dtype=unwrapped_model.dtype, device=self.device
        )
       
        past_key_values = None
        is_finished = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        is_step_finished = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        attention_mask = prompt_mask
        reasoning_step_counter = 0
        
        # --- START: MODIFICATION 1 ---
        # A new state variable to track the end position of the step for each sequence.
        # 각 시퀀스의 스텝 종료 위치를 추적하기 위한 새로운 상태 변수. (`code2`의 `score_position` 역할)
        step_end_positions = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        # --- END: MODIFICATION 1 ---

        model_inputs = unwrapped_model.prepare_inputs_for_generation(prompt_ids, past_key_values=None, attention_mask=attention_mask, return_dict=True)
        outputs = unwrapped_model(**model_inputs, use_cache=True)
        past_key_values = outputs.past_key_values
        next_token_logits = outputs.logits[:, -1, :]

        for cur_len in range(prompt_len, total_len):
            if is_finished.all():
                break
            
            # ... (Sampling logic is the same as before) ...
            
            # --- START: MODIFICATION 2 ---
            # Update the end positions for sequences that just completed a step.
            # 방금 스텝을 완료한 시퀀스들의 종료 위치를 업데이트합니다.
            can_resample_pause = (is_at_delimiter & (cur_len >= prompt_len + self.args.smc_warmup_tokens) & (reasoning_step_counter < self.args.smc_max_resampling_steps))
            step_completion_mask = (can_resample_pause | is_finished) & active_for_generation
            
            # Only update positions that haven't been set yet for this step.
            # 이번 스텝에서 아직 설정되지 않은 위치만 업데이트합니다.
            newly_finished_mask = step_completion_mask & (step_end_positions == 0)
            step_end_positions[newly_finished_mask] = cur_len + 1 #TODO: 왜 cur_len + 1?

            is_step_finished[step_completion_mask] = True
            # --- END: MODIFICATION 2 ---
            
            # ... (The rest of the generation loop is the same)

            if is_step_finished.all():
                sequences, is_finished, attention_mask, past_key_values, all_policy_logits, step_end_positions = self._resample_states(
                    unwrapped_model, sequences, attention_mask, is_finished, 
                    past_key_values, all_policy_logits, step_end_positions, prompt_len, reasoning_step_counter
                )
                is_step_finished = is_finished.clone()
                reasoning_step_counter += 1
                
        completion_ids = sequences[:, prompt_len:]
        return {"completion_ids": completion_ids}


    def _resample_states(self, unwrapped_model, sequences, attention_mask, 
                        is_finished, past_key_values, all_policy_logits,
                        step_end_positions, prompt_len, reasoning_step_counter):
        # --- START: MODIFICATION 3 (Major Refactoring) ---
        # This entire method is refactored to handle variable sequence lengths.
        # 이 메소드 전체가 가변 시퀀스 길이를 처리하도록 리팩토링되었습니다.
        
        batch_size = sequences.shape[0]
        num_groups = batch_size // self.group_size
        
        # 1. Find the max length among all completed steps in this batch.
        # 이 배치에서 완료된 스텝 중 가장 긴 길이를 찾습니다.
        max_len_in_step = torch.max(step_end_positions)
        
        # 2. Prepare padded inputs for efficient, batched forward pass.
        # 효율적인 배치 forward pass를 위해 패딩된 입력을 준비합니다.
        padded_sequences = sequences[:, :max_len_in_step]
        padded_attention_mask = attention_mask[:, :max_len_in_step]
        
        # 3. Create a score_mask to ignore tokens beyond each sequence's actual end position.
        # 각 시퀀스의 실제 종료 위치를 넘어가는 토큰들을 무시하기 위해 score_mask를 생성합니다.
        # Shape: (batch_size, max_completion_len_in_step)
        score_mask = torch.arange(max_len_in_step - prompt_len, device=self.device)[None, :] < (step_end_positions - prompt_len)[:, None]

        # 4. Get logps for the padded batch.
        # 패딩된 배치에 대한 logps를 얻습니다.
        completion_len_for_logits = max_len_in_step - prompt_len
        policy_logits = all_policy_logits[:, :completion_len_for_logits, :]
        logps_policy, logps_ref = self._get_logps(
            unwrapped_model, padded_sequences, padded_attention_mask, completion_len_for_logits, policy_logits
        )

        # 5. Calculate sum_of_ratios using the score_mask to handle variable lengths.
        # score_mask를 사용하여 가변 길이를 처리하고 sum_of_ratios를 계산합니다.
        per_token_ratios = torch.exp(logps_policy - logps_ref)
        sum_of_ratios = (per_token_ratios * score_mask).sum(dim=-1)
        current_scores = self.args.smc_beta * sum_of_ratios

        # ... (Logging logic remains here)
        
        # 6. Standardization and resampling logic is the same.
        # 표준화 및 리샘플링 로직은 동일합니다.
        grouped_scores = current_scores.view(num_groups, self.group_size)
        mean = grouped_scores.mean(dim=-1, keepdim=True)
        std = grouped_scores.std(dim=-1, keepdim=True)
        standardized_scores = (grouped_scores - mean) / (std + 1e-6)
        resampling_log_weights = torch.log(torch.abs(standardized_scores) + 1e-9) * self.args.smc_temperature
        
        if torch.isnan(resampling_log_weights).any() or torch.isinf(resampling_log_weights).any():
            resampling_log_weights = torch.nan_to_num(resampling_log_weights, nan=-1e9, posinf=-1e9, neginf=-1e9)
        
        resampled_indices_local = ordered_stratified_resampling(resampling_log_weights)
        base_indices = torch.arange(0, batch_size, self.group_size, device=self.device).unsqueeze(1)
        final_indices = (resampled_indices_local + base_indices).flatten()

        # 7. Re-index all states.
        # 모든 상태를 리인덱싱합니다.
        sequences = sequences[final_indices]
        is_finished = is_finished[final_indices]
        attention_mask = attention_mask[final_indices]
        past_key_values = tuple(tuple(tensor[final_indices] for tensor in layer) for layer in past_key_values)
        all_policy_logits = all_policy_logits[final_indices]
        
        # 8. Reset step_end_positions for the next reasoning step.
        # 다음 reasoning step을 위해 step_end_positions를 리셋합니다.
        step_end_positions = torch.zeros_like(step_end_positions)

        return sequences, is_finished, attention_mask, past_key_values, all_policy_logits, step_end_positions
        # --- END: MODIFICATION 3 ---

    # ... (The rest of the class methods: _get_logps, _apply_sampling_filters, _check_delimiter)