import torch
from torch import nn
import os
import re
from typing import TYPE_CHECKING
from trl.trainer.utils import selective_log_softmax
import wandb

if TYPE_CHECKING:
    from .smcgrpo_trainer import GRPOTrainer


def ordered_stratified_resampling(weights: torch.Tensor) -> torch.Tensor:
    device = weights.device
    num_groups, num_generations = weights.shape
    probs = weights / torch.sum(weights, dim=-1, keepdim=True)
    u = (torch.arange(num_generations, device=device) + torch.rand(num_groups, num_generations, device=device)) / num_generations
    cumulative_probs = torch.cumsum(probs, dim=-1)
    cumulative_probs[:, -1] += 1e-6
    indices = torch.searchsorted(cumulative_probs, u, right=False)
    return indices.long()


class SMCGenerator:
    def __init__(self, trainer: "GRPOTrainer"):
        self.trainer = trainer
        self.args = trainer.args
        self.tokenizer = trainer.tokenizer
        self.device = trainer.accelerator.device
        self.group_size = self.args.num_generations

        # segment reasoning steps
        self.delimiter_len = 0
        self.delimiter_tensor = None
        if self.args.smc_step_delimiter_string is not None:
            tokens = self.tokenizer.encode(self.args.smc_step_delimiter_string, add_special_tokens=False)
            self.delimiter_len = len(tokens)
            self.delimiter_tensor = torch.tensor(tokens, device=self.device, dtype=torch.long)

        if self.trainer.is_world_process_zero() and "wandb" in self.args.report_to:
            wandb.define_metric("reasoning_step")
            wandb.define_metric("smc/*", step_metric="reasoning_step")
            
    def _initialize_state(self, unwrapped_model: nn.Module, prompt_ids: torch.Tensor, prompt_mask: torch.Tensor):
        batch_size, prompt_len = prompt_ids.shape
        total_len = prompt_len + self.args.max_completion_length
        
        self.prompt_len = prompt_len
        self.sequences = torch.full((batch_size, total_len), self.trainer.pad_token_id, dtype=torch.long, device=self.device) #TODO: intializing full sequences with padding takes a lot of memory?
        self.sequences[:, :prompt_len] = prompt_ids
        
        self.all_policy_logits = torch.empty(
            (batch_size, self.args.max_completion_length, unwrapped_model.config.vocab_size),
            dtype=torch.float32, device=self.device
        )
       
        self.past_key_values = None
        self.is_finished = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        self.was_finished = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        self.is_step_finished = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        self.attention_mask = prompt_mask
        self.reasoning_step_counter = 0
        self.prev_scores = torch.ones(batch_size, dtype=torch.float32, device=self.device)
        self.cur_len = torch.full((batch_size,), prompt_len, dtype=torch.long, device=self.device)
        self.last_delimiter_check_len = torch.full_like(self.cur_len, prompt_len)

    @torch.no_grad()
    def generate(self, unwrapped_model: nn.Module, prompt_ids: torch.Tensor, prompt_mask: torch.Tensor) -> dict:
        self._initialize_state(unwrapped_model, prompt_ids, prompt_mask)
        batch_size = prompt_ids.shape[0]

        model_inputs = unwrapped_model.prepare_inputs_for_generation(prompt_ids, past_key_values=None, attention_mask=self.attention_mask, return_dict=True)
        outputs = unwrapped_model(**model_inputs, use_cache=True) #TODO: prefilling
        self.past_key_values = outputs.past_key_values
        next_token_logits = outputs.logits[:, -1, :]

        while True:
            position_ids = self.cur_len.unsqueeze(-1)
            active_for_generation = ~self.is_step_finished
            
            next_token_logits = self._apply_sampling_filters(next_token_logits)
            
            completion_pos = self.cur_len - self.prompt_len
            self.all_policy_logits[torch.arange(batch_size), completion_pos, :] = next_token_logits.to(torch.float32)
            
            probs = torch.nn.functional.softmax(next_token_logits / self.args.temperature, dim=-1)
            sampled_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            
            next_tokens = torch.full_like(sampled_tokens, self.trainer.pad_token_id)
            next_tokens[active_for_generation] = sampled_tokens[active_for_generation]
            
            self.sequences[torch.arange(batch_size), self.cur_len] = next_tokens
            
            self.is_finished |= (next_tokens == self.trainer.eos_token_id)
            self.is_finished |= (self.cur_len >= self.prompt_len + self.args.max_completion_length - 1)

            if self.is_finished.all():
                break
            
            self.attention_mask = torch.cat([self.attention_mask, torch.ones((batch_size, 1), dtype=torch.long, device=self.device)], dim=1)

            is_at_delimiter = self._check_delimiter(self.sequences, self.cur_len)
            can_resample_pause = (
                is_at_delimiter & 
                (self.cur_len >= self.prompt_len + self.args.smc_warmup_tokens) & 
                (self.reasoning_step_counter < self.args.smc_max_resampling_steps)
            )
            step_completion_mask = (can_resample_pause | self.is_finished) & active_for_generation #TODO: 현재 step에서 generat된거는 제대로 resampling 되도록.

            self.is_step_finished[step_completion_mask] = True
            self.cur_len[~self.is_step_finished] += 1
            
            model_inputs = unwrapped_model.prepare_inputs_for_generation( #TODO: 직접 generation 하지말고 transformers .generate활용
                next_tokens.unsqueeze(-1), 
                past_key_values=self.past_key_values, 
                attention_mask=self.attention_mask, 
                return_dict=True
            ) #TODO: only active trajectory to be generated.
            model_inputs['position_ids'] = position_ids
            outputs = unwrapped_model(**model_inputs, use_cache=True)
            self.past_key_values = outputs.past_key_values
            next_token_logits = outputs.logits[:, -1, :]
            
            if self.is_step_finished.all(): #TODO: Last step일 때도 resampling을 해야하는지? 안해두 될거 같은데..
                self._resample_states(unwrapped_model)
                self.was_finished = self.is_finished.clone()
                self.is_step_finished = self.is_finished.clone()
                self.reasoning_step_counter += 1
                
                if self.is_finished.all():
                    break
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        return self.sequences[:, self.prompt_len:]


    def _apply_sampling_filters(self, logits):
        if self.args.top_k is not None and self.args.top_k > 0:
            top_k = min(self.args.top_k, logits.size(-1))
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = -float("Inf")

        if self.args.top_p is not None and self.args.top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > self.args.top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = -float("Inf")
            
        if self.args.min_p is not None and self.args.min_p > 0.0:
            probs = torch.nn.functional.softmax(logits, dim=-1)
            max_probs = torch.max(probs, dim=-1, keepdim=True)[0]
            threshold = max_probs * self.args.min_p
            indices_to_remove = probs < threshold
            logits[indices_to_remove] = -float("Inf")
        
        return logits
    
    
    def _check_delimiter(self, sequences: torch.Tensor, cur_len: torch.Tensor) -> torch.Tensor: #TODO: need more robust way of catching delimiters
        batch_size = sequences.shape[0]
        delimiter_pattern = re.compile(r"\n\s*\n")
        is_at_delimiter = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        for i in range(batch_size):
            start_idx = self.last_delimiter_check_len[i]
            end_idx = cur_len[i] + 1
            
            if start_idx >= end_idx:
                continue
            
            new_token_segment = sequences[i, start_idx:end_idx]
            decoded_segment = self.tokenizer.decode(new_token_segment, skip_special_tokens=False)
            match = re.search(delimiter_pattern, decoded_segment)
            if match:
                is_at_delimiter[i] = True
                self.last_delimiter_check_len[i] = cur_len[i]
                
        return is_at_delimiter


    def _resample_states(self, unwrapped_model):
        batch_size = self.sequences.shape[0]
        num_groups = batch_size // self.group_size
        
        max_len_in_step = torch.max(self.cur_len)
        padded_sequences = self.sequences[:, :max_len_in_step]
        padded_attention_mask = self.attention_mask[:, :max_len_in_step]
        
        completion_len_for_logits = max_len_in_step - self.prompt_len
        policy_logits = self.all_policy_logits[:, :completion_len_for_logits, :]

        logps_policy, logps_ref = self._get_logps(unwrapped_model, padded_sequences, padded_attention_mask, completion_len_for_logits, policy_logits)
        
        score_mask = torch.arange(completion_len_for_logits, device=self.device)[None, :] < (self.cur_len - self.prompt_len)[:, None]
        
        per_token_ratios = logps_policy - logps_ref
        sum_of_ratios = (per_token_ratios * score_mask).sum(dim=-1)
        current_scores = self.args.smc_beta * sum_of_ratios
        
        grouped_scores = current_scores.view(num_groups, self.group_size)
        
        # Calculate mean/std only on active (not finished) sequences to avoid skew
        active_mask = ~self.was_finished.view(num_groups, self.group_size) #TODO: what if only 1 trajectory is left? Don't do resampling? 
        if active_mask.any():
            mean = grouped_scores[active_mask].mean()
            std = grouped_scores[active_mask].std()
        else:
            mean = 0
            std = 1
            
        standardized_scores = (grouped_scores - mean) / (std + 1e-6)
        standardized_value = torch.nn.functional.softmax(standardized_scores, dim=-1) #TODO: discussion: softmax is ok?
        current_value_processed = standardized_value.flatten() ** self.args.smc_temperature #TODO: produces NaN for temperature smaller than 0.5.
        
        resampling_weights = current_value_processed / self.prev_scores #TODO: log scale?
        resampling_weights[self.was_finished] = 1.0

        resampling_weights_grouped = resampling_weights.view(num_groups, self.group_size)
        resampled_indices_local = ordered_stratified_resampling(resampling_weights_grouped)
        base_indices = torch.arange(0, batch_size, self.group_size, device=self.device).unsqueeze(1)
        final_indices = (resampled_indices_local + base_indices).flatten()

        self.sequences = self.sequences[final_indices]
        self.is_finished = self.is_finished[final_indices]
        self.attention_mask = self.attention_mask[final_indices]
        self.past_key_values = tuple(tuple(tensor[final_indices] for tensor in layer) for layer in self.past_key_values)
        self.all_policy_logits = self.all_policy_logits[final_indices]
        self.prev_scores = current_value_processed[final_indices]
        self.cur_len = self.cur_len[final_indices]
        self.last_delimiter_check_len = self.last_delimiter_check_len[final_indices]
        
        # logging
        if self.trainer.is_world_process_zero() and "wandb" in self.args.report_to:
            wandb.log({
                "smc/per_step_mean_sum_of_ratios": sum_of_ratios.mean().item(),
                "smc/per_step_mean_value": mean.item(),
                "smc/per_step_mean_std": std.item(),
                "smc/mean_standardized_value": standardized_value.flatten()[~self.was_finished].mean().item(),
                "smc/mean_resampling_weight": resampling_weights[~self.was_finished].mean().item(),
                "global_step": self.trainer.state.global_step,
                "reasoning_step": self.reasoning_step_counter
            })

    def _get_logps(self, unwrapped_model, full_sequences, full_attention_mask, completion_len, policy_logits):
        completion_ids = full_sequences[:, -completion_len:]
        logps_policy = selective_log_softmax(policy_logits, completion_ids)
        
        # We only need a forward pass for the reference model
        if self.trainer.ref_model is None:
            with unwrapped_model.disable_adapter():
                ref_outputs = unwrapped_model(input_ids=full_sequences, attention_mask=full_attention_mask, return_dict=True)
        else:
            ref_outputs = self.trainer.ref_model(input_ids=full_sequences, attention_mask=full_attention_mask, return_dict=True)
        
        ref_logits = ref_outputs.logits[:, -completion_len-1:-1, :] #TODO: bfloat16 vs torch.float32?
        logps_ref = selective_log_softmax(ref_logits.to(torch.float32), completion_ids)
        return logps_policy, logps_ref