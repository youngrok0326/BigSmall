import torch
from torch import nn
from typing import TYPE_CHECKING, Dict, Any

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
    
    # State variable to store the cumulative KL divergence.
    cumulative_kl = torch.zeros(batch_size, dtype=torch.float32, device=device)
   
    past_key_values = None
    is_finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
    is_step_finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
    resampling_steps_done = torch.zeros(batch_size, dtype=torch.long, device=device)
    
    attention_mask = prompt_mask
    
    # segmenting reasoning steps
    delimiter_tokens = trainer.processing_class.encode(
        trainer.args.smc_step_delimiter_string,
        add_special_tokens=False
    )
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
        
        # CUMULATIVE KL CALCULATION for the single new token
        # logps_policy = torch.nn.functional.log_softmax(next_token_logits, dim=-1)
        # token_logp_policy = logps_policy.gather(-1, sampled_tokens.unsqueeze(-1)).squeeze(-1)
        
        # ref_model_inputs = unwrapped_model.prepare_inputs_for_generation(
        #     next_tokens.unsqueeze(-1),
        #     past_key_values=past_key_values,
        #     attention_mask=attention_mask  #TODO: correct attention mask?
        # )
        # ref_model_inputs['position_ids'] = position_ids

        # if trainer.ref_model is None:
        #     with unwrapped_model.disable_adapter():
        #         ref_outputs = unwrapped_model(**ref_model_inputs, use_cache=False, return_dict=True) #TODO: in the future, we could reduce this extra forward pass
        # else:
        #     ref_outputs = trainer.ref_model(**ref_model_inputs, use_cache=False, return_dict=True)
        
        # logps_ref = torch.nn.functional.log_softmax(ref_outputs.logits[:, -1, :], dim=-1)
        # token_logp_ref = logps_ref.gather(-1, sampled_tokens.unsqueeze(-1)).squeeze(-1)

        # per_token_kl = token_logp_policy - token_logp_ref
        # cumulative_kl[active_for_generation] += per_token_kl[active_for_generation]

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
            attention_mask=attention_mask)
        model_inputs['position_ids'] = position_ids

        outputs = unwrapped_model(**model_inputs, use_cache=True, return_dict=True)
        past_key_values = outputs.past_key_values
        next_token_logits = outputs.logits[:, -1, :]

        if is_step_finished.all():
            # 1. Get the full sequences for the completed step
            step_end_len = cur_len + 1
            full_sequences = sequences[:, :step_end_len]
            full_attention_mask = attention_mask[:, :step_end_len]
            
            # 2. Get logps for the full sequences from both models
            # This is more efficient as it's one large forward pass instead of many small ones.
            logps_policy, _ = trainer._get_per_token_logps_and_entropies(
                unwrapped_model, full_sequences, full_attention_mask, step_end_len - prompt_len
            )
            if trainer.ref_model is None:
                with unwrapped_model.disable_adapter():
                    logps_ref, _ = trainer._get_per_token_logps_and_entropies(
                        unwrapped_model, full_sequences, full_attention_mask, step_end_len - prompt_len
                    )
            else:
                logps_ref, _ = trainer._get_per_token_logps_and_entropies(
                    trainer.ref_model, full_sequences, full_attention_mask, step_end_len - prompt_len
                )
            
            # 3. Calculate cumulative KL over the entire generated text so far
            completion_mask = full_attention_mask[:, prompt_len:]
            per_token_kl = logps_policy - logps_ref
            cumulative_kl = (per_token_kl * completion_mask).sum(dim=-1)
            
            # The score `V(s_t)` is beta * cumulative_kl.
            current_scores = trainer.args.smc_beta * cumulative_kl
            
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
            cumulative_kl = cumulative_kl[final_indices]
            is_finished = is_finished[final_indices]
            resampling_steps_done = resampling_steps_done[final_indices]
            
            attention_mask = attention_mask[final_indices]
            
            past_key_values = tuple(
                tuple(tensor[final_indices] for tensor in layer) for layer in past_key_values
            )
            
            resampling_steps_done += 1
            is_step_finished = is_finished.clone()
            
    completion_ids = sequences[:, prompt_len:]

    return completion_ids