from collections import deque
from typing import Any, Optional, Union

import torch
import torch.nn.functional as F
import time
from transformers import GenerationConfig, LogitsProcessorList, StoppingCriteriaList
from transformers.generation.logits_process import (
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)
from transformers.generation.utils import GenerateDecoderOnlyOutput, GenerateEncoderDecoderOutput

import wandb
from .utils import ordered_stratified_resampling

def generate(
    model: Any,
    input_ids: torch.LongTensor,
    logits_processor: Optional[LogitsProcessorList] = None,
    stopping_criteria: Optional[StoppingCriteriaList] = None,
    generation_config: Optional[GenerationConfig] = None,
    synced_gpus: bool = False,
    streamer: Optional[Any] = None,
    logging_config: Optional[dict] = None,
    **model_kwargs,
) -> Union[GenerateDecoderOnlyOutput, GenerateEncoderDecoderOutput, torch.LongTensor]:
    """
    Custom decoding with Self-Confidence Guided Sequential Monte Carlo.
    """
    # ========================================================================
    # 1. Configuration & Initialization
    # ========================================================================
    tokenizer = model_kwargs.pop("tokenizer", None)
    batch_size, prompt_len = input_ids.shape[:2]

    if generation_config.max_new_tokens is not None:
        generation_config.max_length = generation_config.max_new_tokens + prompt_len
    if stopping_criteria is None: stopping_criteria = StoppingCriteriaList()
    if logits_processor is None: logits_processor = LogitsProcessorList()
    breakpoint()
    model._prepare_special_tokens(generation_config)
    stopping_criteria = model._get_stopping_criteria(
        generation_config=generation_config, stopping_criteria=stopping_criteria, tokenizer=tokenizer
    )
    logits_processor = model._get_logits_processor(
        generation_config=generation_config, input_ids_seq_length=prompt_len,
        encoder_input_ids=input_ids, prefix_allowed_tokens_fn=None, logits_processor=logits_processor,
    )

    pad_token_id = generation_config.pad_token_id
    if pad_token_id is None and hasattr(generation_config, "_pad_token_tensor"):
        pad_token_id = generation_config._pad_token_tensor
    if pad_token_id is None and hasattr(model.config, "pad_token_id"):
        pad_token_id = model.config.pad_token_id
    if pad_token_id is None and generation_config.eos_token_id is not None:
        pad_token_id = generation_config.eos_token_id
    has_eos_stopping_criteria = any(hasattr(criteria, "eos_token_id") for criteria in stopping_criteria)

    output_attentions = generation_config.output_attentions
    output_hidden_states = generation_config.output_hidden_states
    output_scores = generation_config.output_scores
    output_logits = generation_config.output_logits
    return_dict_in_generate = generation_config.return_dict_in_generate
    do_sample = generation_config.do_sample
    
    output_confidences = getattr(generation_config, "output_confidences", False)
    step_confidences = [] if (return_dict_in_generate and output_confidences) else None 

    scores = () if (return_dict_in_generate and output_scores) else None
    raw_logits = () if (return_dict_in_generate and output_logits) else None
    decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
    cross_attentions = () if (return_dict_in_generate and output_attentions) else None
    decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None
    
    if return_dict_in_generate and model.config.is_encoder_decoder:
        encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
        encoder_hidden_states = model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None

    # --- SMC Configuration ---
    use_smc = getattr(generation_config, "use_smc", False)
    if not use_smc:
        # Fall back to standard sampling if SMC is disabled
        return model.sample(
            input_ids, 
            logits_processor=logits_processor, 
            stopping_criteria=stopping_criteria,
            generation_config=generation_config, 
            synced_gpus=synced_gpus, 
            streamer=streamer, 
            **model_kwargs,
        )

    num_generations = getattr(generation_config, "num_generations", 8)
    num_groups = batch_size // num_generations
    smc_confidence_eta = getattr(generation_config, "smc_confidence_eta", 1.0)
    smc_confidence_window_size = getattr(generation_config, "smc_confidence_window_size", 16)
    smc_topk = getattr(generation_config, "smc_topk", -1)
    step_token = getattr(generation_config, "step_token", "\n\n")
    stop_token = getattr(generation_config, "stop_token", "\\boxed")
    scoring = getattr(generation_config, "scoring", "entropy")
    return_all = getattr(generation_config, "return_all", False)
    
    # --- Logging Initialization ---
    is_main_process = logging_config.get("is_main_process", False) if logging_config else False
    report_to = logging_config.get("report_to", []) if logging_config else []
    global_step = logging_config.get("global_step", 0) if logging_config else 0
    
    smc_table = None
    if is_main_process and "wandb" in report_to:
        columns = [
            "global_step", "resampling_step", "group_index", "confidence",
        ]
        smc_table = wandb.Table(columns=columns)

    unfinished_sequences = torch.ones(batch_size, dtype=torch.bool, device=input_ids.device)
    conf_history = torch.zeros(
        batch_size, smc_confidence_window_size, device=input_ids.device
    )
    age = torch.zeros(batch_size, device=input_ids.device, dtype=torch.long)
    index_ptr = 0

    # ========================================================================
    # 2. Main Generation Loop
    # ========================================================================
    steps = 0
    cur_len = prompt_len
    max_new_tokens = getattr(generation_config, "max_new_tokens", 512)
    step_buf = [""] * batch_size
    step_done = torch.zeros(batch_size, dtype=torch.bool, device=input_ids.device)
    model_kwargs["cache_position"] = torch.arange(cur_len, device=input_ids.device)
 
    warpers = LogitsProcessorList()
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


    while steps < max_new_tokens:
        model_inputs = model.prepare_inputs_for_generation(input_ids, **model_kwargs)
        model_inputs.update({"output_attentions": output_attentions} if output_attentions else {})
        model_inputs.update({"output_hidden_states": output_hidden_states} if output_hidden_states else {})
        torch.cuda.synchronize()
        start = time.time()
        with torch.no_grad():
            outputs = model(**model_inputs, return_dict=True)
            next_token_logits = outputs.logits[:, -1, :].detach()
        torch.cuda.synchronize()
        end = time.time()
        print(f"Step {steps} inference time: {end - start:.4f} seconds")
        if hasattr(outputs, "past_key_values") and outputs.past_key_values is not None:
            model_kwargs["past_key_values"] = outputs.past_key_values

        next_token_scores = logits_processor(input_ids, next_token_logits)
    
        if len(warpers) > 0:
            next_token_scores = warpers(input_ids, next_token_scores)

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
      
        # --- Step A: Propose Next Token ---
        if do_sample:
            probs = F.softmax(next_token_scores, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
        else:
            next_tokens = torch.argmax(next_token_scores, dim=-1)  
        
        if smc_topk > 0:
            k = min(smc_topk, next_token_scores.size(-1))
            logits_for_entropy = torch.topk(next_token_scores, k, dim=-1).values
        else:
            logits_for_entropy = next_token_scores
        probs_for_entropy = F.softmax(logits_for_entropy, dim=-1)
        entropy = torch.distributions.Categorical(probs=probs_for_entropy).entropy()
        step_conf_tensor = torch.exp(-entropy)

        # Confidence Update (freeze paused)
        paused = step_done & unfinished_sequences
        active_mask = unfinished_sequences & (~paused)
        
        if active_mask.any():
            conf_history[active_mask, index_ptr] = step_conf_tensor[active_mask]

        age.add_(active_mask.long()).clamp_(max=smc_confidence_window_size)
        index_ptr = (index_ptr + 1) % smc_confidence_window_size
            
        if step_confidences is not None:
            step_confidences.append(step_conf_tensor.clone())
        
        # Detect step boundary
        if tokenizer is not None and step_token is not None:
            active_mask = unfinished_sequences & (~step_done)
            active_idx = torch.nonzero(active_mask, as_tuple=False).view(-1)
            if active_idx.numel() > 0:
                newly_decoded = tokenizer.batch_decode(next_tokens[active_idx], skip_special_tokens=False)
                new_hits_mask = torch.zeros_like(step_done)
                for idx, decoded_str in zip(active_idx.tolist(), newly_decoded):
                    step_buf[idx] += decoded_str
                    if step_token in step_buf[idx]:
                        new_hits_mask[idx] = True
                        step_buf[idx] = ""
                step_done |= new_hits_mask

        # --- Step C: Update State ---
        if has_eos_stopping_criteria and pad_token_id is not None:
            # Pause rows that completed the step in this iteration
            if 'step_done' in locals():
                paused = step_done & unfinished_sequences
                if paused.any():
                    next_tokens = next_tokens.masked_fill(paused, pad_token_id)
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (~ unfinished_sequences)

        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        if model_kwargs.get("attention_mask") is not None:
            attn = model_kwargs["attention_mask"]
            ones = torch.ones((batch_size, 1), dtype=attn.dtype, device=attn.device)
            model_kwargs["attention_mask"] = torch.cat([attn, ones], dim=-1)
        if streamer: 
            streamer.put(next_tokens.cpu())
        
        standard_stop = stopping_criteria(input_ids, scores)
        if isinstance(standard_stop, torch.Tensor): 
            unfinished_sequences &= ~standard_stop

        # resampling weight
        w = (conf_history.sum(dim=1) / age) ** smc_confidence_eta

        master_indices = torch.arange(batch_size, device=input_ids.device)
        w_reshaped = w.view(num_groups, num_generations)
        unfinished_reshaped = unfinished_sequences.view(num_groups, num_generations)
        step_done_reshaped = step_done.view(num_groups, num_generations) if 'step_done' in locals() else torch.zeros_like(unfinished_reshaped)
        alive = unfinished_reshaped.sum(dim=1)
        remain = unfinished_reshaped & (~step_done_reshaped)
        needs_resampling_mask = (remain.sum(dim=1) == 0) & (alive > 1)

        if needs_resampling_mask.any():
            weights_to_resample = w_reshaped[needs_resampling_mask]
            unfinished_mask_for_resampling = unfinished_reshaped[needs_resampling_mask]
            weights_to_resample[~unfinished_mask_for_resampling] = 0.0
            group_sums = weights_to_resample.sum(dim=1, keepdim=True)
            norm_weights = weights_to_resample / group_sums

            resampled_local_indices = torch.multinomial(
                norm_weights, num_samples=num_generations, replacement=True
            )

            all_indices_reshaped = master_indices.view(num_groups, num_generations)
            parent_pool_indices = all_indices_reshaped[needs_resampling_mask]
            resampled_global_parents = parent_pool_indices.gather(1, resampled_local_indices)

            new_parent_candidates = master_indices.view(num_groups, num_generations).clone()
            new_parent_candidates[needs_resampling_mask] = resampled_global_parents
            new_parent_candidates = new_parent_candidates.flatten()

            master_indices[unfinished_sequences] = new_parent_candidates[unfinished_sequences]

            input_ids = input_ids.index_select(0, master_indices)
            unfinished_sequences = unfinished_sequences.index_select(0, master_indices)
            if 'step_done' in locals():
                step_done = step_done.index_select(0, master_indices)
                step_buf = [step_buf[j] for j in master_indices.tolist()]
            conf_history = conf_history.index_select(0, master_indices)
            age = age.index_select(0, master_indices)
            if "attention_mask" in model_kwargs:
                model_kwargs["attention_mask"] = model_kwargs["attention_mask"].index_select(0, master_indices)
            if "past_key_values" in model_kwargs and model_kwargs["past_key_values"]:
                model_kwargs["past_key_values"] = tuple(
                    tuple(p.index_select(0, master_indices) for p in layer_past)
                    for layer_past in model_kwargs["past_key_values"]
                )
            # Reset step buffers/flags for resampled groups
            if 'step_done' in locals():
                reset_mask = needs_resampling_mask.view(num_groups, 1).expand(num_groups, num_generations).reshape(-1)
                for j in torch.nonzero(reset_mask, as_tuple=False).view(-1).tolist():
                    step_done[j] = False
                    step_buf[j] = ""

        # Update loop counters
        cur_len += 1
        steps += 1
        model_kwargs["cache_position"] = model_kwargs["cache_position"][-1:] + 1

        del outputs

        if unfinished_sequences.max() == 0 and not synced_gpus:
            break

    # ========================================================================
    # 3. Finalization & Return
    # ========================================================================
    if streamer: 
        streamer.end()
        
    if smc_table is not None:
        wandb.log({"smc_table": smc_table})
    
    if return_dict_in_generate:
        confidences_tensor = None
        if step_confidences is not None and len(step_confidences) > 0:
            confidences_tensor = torch.stack(step_confidences, dim=0).transpose(0, 1)
        if model.config.is_encoder_decoder:
            output = GenerateEncoderDecoderOutput(
                sequences=input_ids, scores=scores, logits=raw_logits,
                encoder_attentions=encoder_attentions, encoder_hidden_states=encoder_hidden_states,
                decoder_attentions=decoder_attentions, cross_attentions=cross_attentions,
                decoder_hidden_states=decoder_hidden_states, past_key_values=model_kwargs.get("past_key_values"),
            )
        else:
            output = GenerateDecoderOnlyOutput(
                sequences=input_ids, scores=scores, logits=raw_logits,
                attentions=decoder_attentions, hidden_states=decoder_hidden_states,
                past_key_values=model_kwargs.get("past_key_values"),
            )
        output.smc_log_weights = w
        if confidences_tensor is not None:
            output.smc_confidences = confidences_tensor
        return output
    else:
        return input_ids
