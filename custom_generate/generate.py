from collections import deque
from typing import Any, Optional, Union

import torch
import torch.nn.functional as F

from transformers import GenerationConfig, LogitsProcessorList, StoppingCriteriaList
from transformers.generation.logits_process import (
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)
from transformers.generation.utils import GenerateDecoderOnlyOutput, GenerateEncoderDecoderOutput

import re
import wandb
from .utils import ordered_stratified_resampling
from trl.trainer.utils import selective_log_softmax


def generate(
    model: Any,
    input_ids: torch.LongTensor,
    logits_processor: Optional[LogitsProcessorList] = None,
    stopping_criteria: Optional[StoppingCriteriaList] = None,
    generation_config: Optional[GenerationConfig] = None,
    synced_gpus: bool = False,
    streamer: Optional[Any] = None,
    ref_model: Optional[Any] = None,
    logging_config: Optional[dict] = None,
    **model_kwargs,
) -> Union[GenerateDecoderOnlyOutput, GenerateEncoderDecoderOutput, torch.LongTensor]:
    """Custom decoding with Sequential Monte Carlo.
    Args:
        model: PreTrainedModel with a LM head.
        input_ids: Prompt ids of shape (batch, seq_len).
        logits_processor: Optional logits processors.
        stopping_criteria: Optional stopping criteria.
        generation_config: GenerationConfig controlling sampling/outputs.
        synced_gpus: Keep looping to max length for distributed setups.
        streamer: Optional streamer for incremental tokens.
        ref_model: reference model. If None, use the current model as reference with adapters disabled.
        logging_config: Optional logging configuration.
        **model_kwargs: Forward pass kwargs (e.g., attention_mask).
    Returns:
        GenerateDecoderOnlyOutput, GenerateEncoderDecoderOutput, or LongTensor
        depending on `return_dict_in_generate` and model type.
    """
    #TODO: Get SMC parameters from generation_config or set defaults #should we pass with /self.generation_config/ or /self.args/? # if with generation config, update it to also have the self.args in the trainer.
    num_generations = getattr(generation_config, "num_generations", 8)
    use_smc = getattr(generation_config, "use_smc", False)
    tokenizer = model_kwargs.pop("tokenizer", None)
    
    batch_size, prompt_len = input_ids.shape[:2]
    num_groups = batch_size // num_generations
    
    if generation_config.max_new_tokens is not None:
        generation_config.max_length = generation_config.max_new_tokens + prompt_len
    if stopping_criteria is None:
        stopping_criteria = StoppingCriteriaList()
    if logits_processor is None:
        logits_processor = LogitsProcessorList()
        
    model._prepare_special_tokens(generation_config)
    
    stopping_criteria = model._get_stopping_criteria(
        generation_config=generation_config, 
        stopping_criteria=stopping_criteria, 
        tokenizer=tokenizer
    )
     
    logits_processor = model._get_logits_processor(
        generation_config=generation_config,
        input_ids_seq_length=prompt_len,
        encoder_input_ids=input_ids,
        prefix_allowed_tokens_fn=None,
        logits_processor=logits_processor,
    )
    
    # If DeepCONF is not enabled, fall back to standard sampling
    if not use_smc:
        return model._sample(
            input_ids,
            logits_processor=logits_processor,
            stopping_criteria=stopping_criteria,
            generation_config=generation_config,
            synced_gpus=synced_gpus,
            streamer=streamer,
            **model_kwargs,
        )
    smc_beta = getattr(generation_config, "smc_beta", 1.0)
    smc_temperature = getattr(generation_config, "smc_temperature", 1.0)
    smc_step_delimiter_string = r"\n\s*\n" #getattr(generation_config, "smc_step_delimiter_string", "\n\n")
    smc_warmup_tokens = getattr(generation_config, "smc_warmup_tokens", 10)
    smc_max_resampling_steps = getattr(generation_config, "smc_max_resampling_steps", 5)
    #TODO: Modification end
        
    # Initialize values
    # Handle pad token properly (following HF best practices)
    pad_token_id = generation_config.pad_token_id
    if pad_token_id is None and hasattr(generation_config, "_pad_token_tensor"):
        pad_token_id = generation_config._pad_token_tensor
    if pad_token_id is None and hasattr(model.config, "pad_token_id"):
        pad_token_id = model.config.pad_token_id
    if pad_token_id is None and generation_config.eos_token_id is not None:
        # Use eos token as pad token if not set
        pad_token_id = generation_config.eos_token_id
        
    output_attentions = generation_config.output_attentions
    output_hidden_states = generation_config.output_hidden_states
    output_scores = generation_config.output_scores
    output_logits = generation_config.output_logits
    return_dict_in_generate = generation_config.return_dict_in_generate
    has_eos_stopping_criteria = any(hasattr(criteria, "eos_token_id") for criteria in stopping_criteria)
    do_sample = generation_config.do_sample
    
    # Initialize attention / hidden states / scores tuples
    scores = () if (return_dict_in_generate and output_scores) else None
    raw_logits = () if (return_dict_in_generate and output_logits) else None
    decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
    cross_attentions = () if (return_dict_in_generate and output_attentions) else None
    decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

    # If model is an encoder-decoder, retrieve encoder attention weights and hidden states
    if return_dict_in_generate and model.config.is_encoder_decoder:
        encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
        encoder_hidden_states = model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None

    #TODO: SMC State variables
    smc_step_finished = torch.zeros(batch_size, dtype=torch.bool, device=input_ids.device)
    reasoning_step_counter = 0
    delimiter_pattern = re.compile(smc_step_delimiter_string) if smc_step_delimiter_string else None
    last_delimiter_check_len = torch.full((batch_size,), prompt_len, dtype=torch.long, device=input_ids.device)
    unfinished_sequences = torch.ones(batch_size, dtype=torch.bool, device=input_ids.device)
    
    avg_logps_policy = torch.zeros(batch_size, device=input_ids.device, dtype=torch.float32)
    prev_scores = torch.ones(batch_size, device=input_ids.device, dtype=torch.float32)
    
    # logging 
    is_main_process = logging_config.get("is_main_process", False) if logging_config else False
    report_to = logging_config.get("report_to", []) if logging_config else []
    global_step = logging_config.get("global_step", 0) if logging_config else 0

    #TODO: Efficient group-wise Prefill Step
    """ # --- SMC: KV Cache Sharing Optimization (Prefill Step) ---
        # 1. Select unique prompts for a single forward pass
        unique_prompt_indices = torch.arange(0, batch_size, num_generations, device=input_ids.device)
        unique_input_ids = input_ids[unique_prompt_indices]
        unique_attention_mask = model_kwargs["attention_mask"][unique_prompt_indices]
        
        # 2. Perform a single forward pass on unique prompts to get the initial KV cache
        prefill_inputs = model.prepare_inputs_for_generation(
            unique_input_ids, attention_mask=unique_attention_mask
        )
        with torch.no_grad():
            prefill_outputs = model(**prefill_inputs, return_dict=True, use_cache=True)
        
        # 3. Get logits for the first token and the initial KV cache
        initial_past_key_values = prefill_outputs.past_key_values
        next_token_logits = prefill_outputs.logits[:, -1, :].detach()

        # 4. Expand (broadcast) the KV cache and logits for all particles
        expanded_past_key_values = tuple(
            tuple(tensor.repeat_interleave(num_generations, dim=0) for tensor in layer)
            for layer in initial_past_key_values
        )
        model_kwargs["past_key_values"] = expanded_past_key_values
        next_token_logits = next_token_logits.repeat_interleave(num_generations, dim=0) """


    # Main generation loop using public controls
    steps = 0
    cur_len = prompt_len
    max_new_tokens = getattr(generation_config, "max_new_tokens", None)
    # Initialize cache_position for first forward over the full prompt
    # Subsequent steps will pass a single position incrementally
    model_kwargs["cache_position"] = torch.arange(cur_len, device=input_ids.device)
    while steps < max_new_tokens and unfinished_sequences.max() != 0:
        ###Modification###
        # --- SMC: Resampling block ---
        active_for_resampling = ~smc_step_finished & unfinished_sequences.bool() #TODO: shouldn't do the resampling for the last reasoning step, but only obtain the weight.
        if active_for_resampling.sum() == 0 and reasoning_step_counter < smc_max_resampling_steps: #TODO: what if only 1 trajectory is left? Don't do resampling? 
            """ with torch.no_grad():
                if ref_model is None:
                    with model.disable_adapter():
                        ref_outputs = model(input_ids=input_ids, attention_mask=model_kwargs["attention_mask"])
                else:
                    ref_outputs = ref_model(input_ids=input_ids, attention_mask=model_kwargs["attention_mask"])

            ref_logits = ref_outputs.logits[:, prompt_len-1:-1, :]
            logps_ref = selective_log_softmax(ref_logits.to(torch.float32), input_ids[:, prompt_len:])
            sum_logps_ref = logps_ref.sum(dim=-1)

            # 3. Calculate scores and resampling weights
            sum_of_ratios = sum_logps_policy - sum_logps_ref #TODO: to be precise. V(s_t) is computed by rewards upto t-1.
            current_scores = smc_beta * sum_of_ratios """
            
            current_scores = -avg_logps_policy #TODO: self-confidence
            grouped_scores = current_scores.view(num_groups, num_generations)
            # resampling_weights = F.softmax(grouped_scores / smc_temperature, dim=-1) #TODO: discussion: softmax is ok? But, we will modify to self-confidence anyway.

            #TODO: standardization & increment weight
            active_mask = unfinished_sequences.view(num_groups, num_generations)
            mean = grouped_scores[active_mask].mean()
            std = grouped_scores[active_mask].std() if active_mask.sum() > 1 else 0.0
            
            standardized_scores = (grouped_scores - mean) / (std + 1e-6)
            standardized_value = F.softmax(standardized_scores, dim=-1)
            current_value_processed = standardized_value.flatten()
            
            # Incremental weight calculation
            resampling_weights = current_value_processed / prev_scores
            # Finished trajectories have a neutral incremental weight of 1.0
            resampling_weights[~unfinished_sequences] = 1.0 
            
            # 4. Resample indices
            resampling_weights_grouped = resampling_weights.view(num_groups, num_generations)
            resampled_indices_local = ordered_stratified_resampling(resampling_weights_grouped)
            base_indices = torch.arange(0, batch_size, num_generations, device=input_ids.device).unsqueeze(1)
            final_indices = (resampled_indices_local + base_indices).flatten()
            
            # 5. Re-index all state tensors
            input_ids = input_ids[final_indices]
            unfinished_sequences = unfinished_sequences[final_indices]
            if "attention_mask" in model_kwargs:
                model_kwargs["attention_mask"] = model_kwargs["attention_mask"][final_indices]
            if "past_key_values" in model_kwargs and model_kwargs["past_key_values"] is not None:
                model_kwargs["past_key_values"] = tuple(
                    tuple(tensor[final_indices] for tensor in layer) for layer in model_kwargs["past_key_values"]
                )
            avg_logps_policy = avg_logps_policy[final_indices]
            
            # 6. Reset states for the next reasoning step
            smc_step_finished.fill_(False)
            reasoning_step_counter += 1
            
            #TODO: wandb
            if is_main_process and "wandb" in report_to:
                wandb.log({
                    "smc/per_step_mean_avg_logps_policy": avg_logps_policy.mean().item(),
                    "smc/per_step_mean_value": mean,
                    "smc/per_step_mean_std": std,
                    "smc/mean_standardized_value": standardized_value.flatten()[unfinished_sequences].mean().item(),
                    "smc/mean_resampling_weight": resampling_weights[unfinished_sequences].mean().item(),
                    "global_step": global_step,
                    "reasoning_step": reasoning_step_counter
                })
            
        ###Modification###
        
        #TODO: if steps > 0: # This is necessary if we do the prefill broadcasting.
        # Prepare model inputs (proper KV cache handling)
        model_inputs = model.prepare_inputs_for_generation(input_ids, **model_kwargs)
        
        # Prepare variable output controls
        model_inputs.update({"output_attentions": output_attentions} if output_attentions else {})
        model_inputs.update({"output_hidden_states": output_hidden_states} if output_hidden_states else {})
        
        # Forward pass with proper KV cache handling
        with torch.no_grad():
            outputs = model(**model_inputs, return_dict=True) #TODO: only the active sequences
            next_token_logits = outputs.logits[:, -1, :].detach()
            
        # Update model kwargs for next iteration (public): carry past_key_values
        if hasattr(outputs, "past_key_values") and outputs.past_key_values is not None:
            model_kwargs["past_key_values"] = outputs.past_key_values

        # Pre-process distribution with logits processors
        next_token_scores = logits_processor(input_ids, next_token_logits)
        
        # Apply logits warpers (e.g., temperature, top-k, top-p) from generation_config
        warpers = LogitsProcessorList()
        # Temperature
        temperature = getattr(generation_config, "temperature", 1.0)
        if temperature is not None and temperature != 1.0:
            warpers.append(TemperatureLogitsWarper(temperature))
        # Top-k
        top_k = getattr(generation_config, "top_k", None)
        if top_k is not None and isinstance(top_k, int) and top_k > 0:
            warpers.append(TopKLogitsWarper(top_k))
        # Top-p
        top_p = getattr(generation_config, "top_p", None)
        if top_p is not None and top_p < 1.0:
            warpers.append(TopPLogitsWarper(top_p))
        if len(warpers) > 0:
            next_token_scores = warpers(input_ids, next_token_scores)

        # Store scores, attentions and hidden_states when required
        if return_dict_in_generate:
            if output_scores:
                scores += (next_token_scores,)
            if output_logits:
                raw_logits += (next_token_logits,)
            if output_attentions:
                decoder_attentions += (
                    (outputs.decoder_attentions,) if model.config.is_encoder_decoder else (outputs.attentions,)
                )
                if model.config.is_encoder_decoder:
                    cross_attentions += (outputs.cross_attentions,)

            if output_hidden_states:
                decoder_hidden_states += (
                    (outputs.decoder_hidden_states,) if model.config.is_encoder_decoder else (outputs.hidden_states,)
                )

        # Token selection
        if do_sample:
            probs = F.softmax(next_token_scores, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
        else:
            next_tokens = torch.argmax(next_token_scores, dim=-1)
            
        ###Modification###
        #TODO: SMC
        log_probs = F.log_softmax(next_token_scores, dim=-1)
        log_probs = torch.gather(log_probs, dim=1, index=next_tokens.unsqueeze(-1)).squeeze(-1) #TODO: self-confidence
        avg_logps_policy = torch.where(
            unfinished_sequences.bool(), (avg_logps_policy * steps + log_probs) / (steps + 1), avg_logps_policy
        )
        ###Modification###
        
        # Finished sentences should have their next token be a padding token
        if has_eos_stopping_criteria and pad_token_id is not None:
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (~unfinished_sequences)
            
        # Update generated ids, model inputs, and length for next step
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        # Update attention mask if available
        if model_kwargs.get("attention_mask") is not None:
            attn = model_kwargs["attention_mask"]
            model_kwargs["attention_mask"] = torch.cat(
                [attn, torch.ones((batch_size, 1), dtype=attn.dtype, device=attn.device)], dim=-1
            )
        # Update cache_position for next step (single next token)
        model_kwargs["cache_position"] = model_kwargs["cache_position"][-1:] + 1
        if streamer is not None:
            streamer.put(next_tokens.cpu())
            
        # Update unfinished sequences with standard stopping criteria (per-sequence if available)
        sc = stopping_criteria(input_ids, scores)
        if isinstance(sc, torch.Tensor):
            unfinished_sequences = unfinished_sequences & ~sc
        elif sc:
            # global stop
            unfinished_sequences = torch.zeros_like(unfinished_sequences)

        #TODO: Apply SMC stopping        
        ###Modification###
        # Check for SMC step completion via delimiter
        if delimiter_pattern and cur_len >= prompt_len + smc_warmup_tokens: #TODO: how to robustly segment each reasoning step.
            for i in range(batch_size):
                if not smc_step_finished[i] and unfinished_sequences[i]:        
                    start_idx = last_delimiter_check_len[i]
                    end_idx = cur_len + 1

                    new_token_segment = input_ids[i, start_idx:end_idx]
                    decoded_segment = tokenizer.decode(new_token_segment, skip_special_tokens=False) #TODO: temporary
                    
                    if delimiter_pattern.search(decoded_segment):
                        smc_step_finished[i] = True
                        last_delimiter_check_len[i] = end_idx
        
        # A sequence's step is also finished if the whole sequence is done
        smc_step_finished |= ~unfinished_sequences.bool() #TODO: incorrect
        ###Modification###
        
        # Early break if all sequences finished and not synchronized
        if unfinished_sequences.max() == 0 and not synced_gpus:
            break
        cur_len += 1
        steps += 1
        
        # Clean up outputs to save memory
        del outputs
        
    if streamer is not None:
        streamer.end()

    # Return results
    if return_dict_in_generate:
        #TODO: smc tensors
        if model.config.is_encoder_decoder:
            output = GenerateEncoderDecoderOutput(
                sequences=input_ids,
                scores=scores,
                logits=raw_logits,
                encoder_attentions=encoder_attentions,
                encoder_hidden_states=encoder_hidden_states,
                decoder_attentions=decoder_attentions,
                cross_attentions=cross_attentions,
                decoder_hidden_states=decoder_hidden_states,
                past_key_values=model_kwargs.get("past_key_values"),
            )
            #TODO: smc tensors
            return output
        else:
            output = GenerateDecoderOnlyOutput(
                sequences=input_ids,
                scores=scores,
                logits=raw_logits,
                attentions=decoder_attentions,
                hidden_states=decoder_hidden_states,
                past_key_values=model_kwargs.get("past_key_values"),
            )
            #TODO: smc tensors
            return output
    else:
        return input_ids