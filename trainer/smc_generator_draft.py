import torch
from torch import nn
from typing import TYPE_CHECKING, Dict, Any

import wandb

if TYPE_CHECKING:
    from .smcgrpo_trainer import GRPOTrainer


def ordered_stratified_resampling(log_weights: torch.Tensor) -> torch.Tensor:
    device = log_weights.device
    num_groups, num_generations = log_weights.shape
    probs = torch.nn.functional.softmax(log_weights, dim=-1)
    u = (torch.arange(num_generations, device=device) + torch.rand(num_groups, num_generations, device=device)) / num_generations
    cumulative_probs = torch.cumsum(probs, dim=-1)
    cumulative_probs[:, -1] += 1e-6
    indices = torch.searchsorted(cumulative_probs, u, right=False)
    return indices.long()


# TODO: wandb에 self-confidence 기록

@torch.no_grad()
def generate_completions_smc(
    trainer: "GRPOTrainer",
    unwrapped_model: nn.Module,
    prompt_ids: torch.Tensor,
    prompt_mask: torch.Tensor
) -> torch.Tensor:

    device = trainer.accelerator.device
    batch_size, prompt_len = prompt_ids.shape
    total_len = prompt_len + trainer.max_completion_length
    group_size = trainer.num_generations
    num_groups = batch_size // group_size

    sequences = torch.full((batch_size, total_len), trainer.pad_token_id, dtype=torch.long, device=device)
    sequences[:, :prompt_len] = prompt_ids
    
    all_policy_logits = torch.empty(
        (batch_size, trainer.max_completion_length, unwrapped_model.config.vocab_size),
        dtype=unwrapped_model.dtype,
        device=device
    )
   
    past_key_values = None
    is_finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
    is_step_finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
    resampling_steps_done = torch.zeros(batch_size, dtype=torch.long, device=device)
    
    attention_mask = prompt_mask
    
    # segmenting reasoning steps
    delimiter_len = 0
    reasoning_step_counter = 0
    if trainer.args.smc_step_delimiter_string is not None:
        delimiter_tokens = trainer.processing_class.encode(
            trainer.args.smc_step_delimiter_string,
            add_special_tokens=False)
        delimiter_len = len(delimiter_tokens)
        delimiter_tensor = torch.tensor(delimiter_tokens, device=device, dtype=torch.long)

    model_inputs = unwrapped_model.prepare_inputs_for_generation(prompt_ids, past_key_values=None, attention_mask=prompt_mask)
    outputs = unwrapped_model(**model_inputs, use_cache=True)
    past_key_values = outputs.past_key_values
    next_token_logits = outputs.logits[:, -1, :]

    for cur_len in range(prompt_len, total_len):
        if is_finished.all():
            break
        
        position_ids = torch.tensor([[cur_len]], dtype=torch.long, device=device).expand(batch_size, -1)
        active_for_generation = ~is_step_finished
        
        # repetition penalty #TODO: vectorized way
        if trainer.args.repetition_penalty != 1.0:
            generated_so_far = sequences[:, prompt_len:cur_len]
            score = torch.gather(next_token_logits, 1, generated_so_far)
            score = torch.where(score < 0, score * trainer.args.repetition_penalty, score / trainer.args.repetition_penalty)
            next_token_logits.scatter_(1, generated_so_far, score)
                
        # Apply Top-K and Top-P filtering
        # Get the top k logits and set the rest to -inf
        if trainer.args.top_k is not None and trainer.args.top_k > 0:
            top_k = min(trainer.args.top_k, next_token_logits.size(-1))  # Safety check
            indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
            next_token_logits[indices_to_remove] = -float("Inf")

        # Get the top p logits and set the rest to -inf
        if trainer.args.top_p is not None and trainer.args.top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
            cumulative_probs = torch.cumsum(torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > trainer.args.top_p
            # Shift the indices so we keep the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            # Scatter the -inf values back to the original logit positions
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            next_token_logits[indices_to_remove] = -float("Inf")
            
        # Apply Min-P filtering
        if trainer.args.min_p is not None and trainer.args.min_p > 0.0:
            # Convert logits to probabilities to perform the check
            probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
            # Find the probability of the most likely token
            max_probs = torch.max(probs, dim=-1, keepdim=True)[0]
            # Set the threshold and remove tokens below it
            threshold = max_probs * trainer.args.min_p
            indices_to_remove = probs < threshold
            next_token_logits[indices_to_remove] = -float("Inf")
            
        completion_pos = cur_len - prompt_len
        all_policy_logits[:, completion_pos, :] = next_token_logits
        
        # Apply Temperature and Sample
        probs = torch.nn.functional.softmax(next_token_logits / trainer.temperature, dim=-1)
        sampled_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
        
        next_tokens = torch.full_like(sampled_tokens, trainer.pad_token_id)
        next_tokens[active_for_generation] = sampled_tokens[active_for_generation]
        
        sequences[active_for_generation, cur_len] = next_tokens[active_for_generation]
        is_finished |= (next_tokens == trainer.eos_token_id)
        
        attention_mask = torch.cat([
            attention_mask,
            torch.ones((batch_size, 1), dtype=torch.long, device=device)
        ], dim=1)

        # segment the reasoning steps
        is_at_delimiter = torch.zeros(batch_size, dtype=torch.bool, device=device)
        if delimiter_len > 0 and cur_len >= prompt_len + delimiter_len - 1:
            last_n_tokens = sequences[:, cur_len - delimiter_len + 1 : cur_len + 1]
            is_at_delimiter = torch.all(last_n_tokens == delimiter_tensor, dim=1)
        
        can_resample_pause = (
            is_at_delimiter &
            (cur_len >= prompt_len + trainer.args.smc_warmup_tokens) &
            (resampling_steps_done < trainer.args.smc_max_resampling_steps)
        )
        
        step_completion_mask = (can_resample_pause | is_finished) & active_for_generation
        is_step_finished[step_completion_mask] = True
        
        model_inputs = unwrapped_model.prepare_inputs_for_generation(
            next_tokens.unsqueeze(-1),
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            return_dict=True)
        model_inputs['position_ids'] = position_ids
        outputs = unwrapped_model(**model_inputs, use_cache=True)
        past_key_values = outputs.past_key_values
        next_token_logits = outputs.logits[:, -1, :]

        if is_step_finished.all():
            step_end_len = cur_len + 1
            full_sequences = sequences[:, :step_end_len]
            full_attention_mask = attention_mask[:, :step_end_len]
            completion_len = step_end_len - prompt_len
            
            # 1. Get policy logits from our stored tensor.
            policy_logits = all_policy_logits[:, :completion_len, :]
            
            # 2. Run forward pass for the reference model ONLY.
            # We use the _get_logps_fallback helper as it contains all necessary logic.
            if 'selective_log_softmax' not in locals(): from trl.trainer.utils import selective_log_softmax
            if '_get_logps_fallback' not in locals():
                @torch.no_grad()
                def _get_logps_fallback(model, sequences, attention_mask, completion_len):
                    outputs = model(input_ids=sequences, attention_mask=attention_mask, return_dict=True)
                    logits, completion_ids = outputs.logits[:, -completion_len-1:-1], sequences[:, -completion_len:]
                    return selective_log_softmax(logits, completion_ids)

            if trainer.ref_model is None:
                with unwrapped_model.disable_adapter():
                    logps_ref = _get_logps_fallback(unwrapped_model, full_sequences, full_attention_mask, completion_len)
            else:
                logps_ref = _get_logps_fallback(trainer.ref_model, full_sequences, full_attention_mask, completion_len)
            
            # 3. Calculate policy logps from the stored logits.
            completion_ids = full_sequences[:, -completion_len:]
            logps_policy = selective_log_softmax(policy_logits, completion_ids)
            
            # 4. The rest of the scoring logic is now efficient.
            per_token_ratios = torch.exp(logps_policy - logps_ref)
            completion_mask = full_attention_mask[:, -completion_len:]
            sum_of_ratios = (per_token_ratios * completion_mask).sum(dim=-1)
            current_scores = trainer.args.smc_beta * sum_of_ratios
            
            # Reshape for group-wise standardization.
            grouped_scores = current_scores.view(num_groups, group_size)
            mean = grouped_scores.mean(dim=-1, keepdim=True)
            std = grouped_scores.std(dim=-1, keepdim=True)
            standardized_scores = (grouped_scores - mean) / (std + 1e-6)
            
            # To implement sqrt(V), calculate log(V) and multiply by temperature (0.5).
            resampling_log_weights = torch.log(torch.abs(standardized_scores) + 1e-9) * trainer.args.smc_temperature
                
            resampled_indices_local = ordered_stratified_resampling(resampling_log_weights)
            base_indices = torch.arange(0, batch_size, group_size, device=device).unsqueeze(1)
            final_indices = (resampled_indices_local + base_indices).flatten()

            sequences = sequences[final_indices]
            is_finished = is_finished[final_indices]
            resampling_steps_done = resampling_steps_done[final_indices]
            attention_mask = attention_mask[final_indices]
            past_key_values = tuple(
                tuple(tensor[final_indices] for tensor in layer) for layer in past_key_values
            )
            all_policy_logits = all_policy_logits[final_indices]
            
            resampling_steps_done += 1
            is_step_finished = is_finished.clone()
            
            # logging
            if trainer.is_world_process_zero() and "wandb" in trainer.args.report_to:
                wandb.define_metric("reasoning_step")
                wandb.define_metric("smc/*", step_metric="reasoning_step")
                wandb.log({
                    "smc/per_step_sum_of_ratios": sum_of_ratios.mean().item(),
                    "global_step": trainer.state.global_step,
                    "reasoning_step": reasoning_step_counter
                })
            reasoning_step_counter += 1
    completion_ids = sequences[:, prompt_len:]

    return completion_ids