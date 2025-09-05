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
    smc_ess_threshold = getattr(generation_config, "smc_ess_threshold", 0.5)
    smc_confidence_window_size = getattr(generation_config, "smc_confidence_window_size", 16)
    
    # --- Logging Initialization ---
    is_main_process = logging_config.get("is_main_process", False) if logging_config else False
    report_to = logging_config.get("report_to", []) if logging_config else []
    global_step = logging_config.get("global_step", 0) if logging_config else 0
    
    smc_table = None
    resampling_step_counter = 0
    if is_main_process and "wandb" in report_to:
        columns = [
            "global_step", "resampling_step", "group_index", "ess",
            "conf_mean", "conf_std",
            "avg_conf_mean", "avg_conf_std",
            "weight_mean", "weight_std"
        ]
        smc_table = wandb.Table(columns=columns)

    # --- SMC State Initialization ---
    log_w = torch.zeros(batch_size, device=input_ids.device, dtype=torch.float32)
    conf_deques = [deque(maxlen=smc_confidence_window_size) for _ in range(batch_size)]
    conf_sums = [0.0 for _ in range(batch_size)]
    unfinished_sequences = torch.ones(batch_size, dtype=torch.bool, device=input_ids.device)

    # ========================================================================
    # 2. Main Generation Loop
    # ========================================================================
    steps = 0
    cur_len = prompt_len
    max_new_tokens = getattr(generation_config, "max_new_tokens", 512)
    model_kwargs["cache_position"] = torch.arange(cur_len, device=input_ids.device)

    while steps < max_new_tokens:
        model_inputs = model.prepare_inputs_for_generation(input_ids, **model_kwargs)
        model_inputs.update({"output_attentions": output_attentions} if output_attentions else {})
        model_inputs.update({"output_hidden_states": output_hidden_states} if output_hidden_states else {})
        
        with torch.no_grad():
            outputs = model(**model_inputs, return_dict=True)
            next_token_logits = outputs.logits[:, -1, :].detach()

        if hasattr(outputs, "past_key_values") and outputs.past_key_values is not None:
            model_kwargs["past_key_values"] = outputs.past_key_values

        next_token_scores = logits_processor(input_ids, next_token_logits)
        
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

        # --- Step B: Calculate & Track Per-Token Confidence ---
        logprobs = F.log_softmax(next_token_scores, dim=-1)
        candidate_mask = torch.isfinite(next_token_scores) 
        
        step_conf_values = [0.0] * batch_size 
        for i in range(batch_size):
            if not unfinished_sequences[i]: continue
            num_candidates = int(candidate_mask[i].sum().item())
            if num_candidates <= 1: conf = 0.0
            else:
                total_lp = torch.sum(logprobs[i][candidate_mask[i]])
                selected_lp = logprobs[i, next_tokens[i]] if candidate_mask[i, next_tokens[i]] else 0.0
                conf = -((total_lp - selected_lp) / (num_candidates - 1)).item()
            
            if len(conf_deques[i]) >= smc_confidence_window_size: 
                conf_sums[i] -= conf_deques[i][0]
            conf_deques[i].append(conf)
            conf_sums[i] += conf
            step_conf_values[i] = conf
        
        if step_confidences is not None:
            step_confidences.append(torch.tensor(step_conf_values, device=input_ids.device))

        # --- Step C: Update State ---
        if has_eos_stopping_criteria and pad_token_id is not None:
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (~ unfinished_sequences)
        
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        if model_kwargs.get("attention_mask") is not None:
            attn = model_kwargs["attention_mask"]
            model_kwargs["attention_mask"] = torch.cat([attn, torch.ones((batch_size, 1), dtype=attn.dtype, device=attn.device)], dim=-1)
        if streamer: 
            streamer.put(next_tokens.cpu())
        
        standard_stop = stopping_criteria(input_ids, scores)
        if isinstance(standard_stop, torch.Tensor): 
            unfinished_sequences &= ~standard_stop

        # --- Step D: WEIGH & RESAMPLE BLOCK (runs every token) ---
        avg_conf_values = (torch.tensor([conf_sums[i] / len(conf_deques[i])
                                         if unfinished_sequences[i] 
                                         else 0.0 
                                         for i in range(batch_size)], device=input_ids.device)
        )
        log_w += smc_confidence_eta * avg_conf_values #TODO: annealing? #TODO: turn into a sqrt of value (sum of reward minus kl term)

        for g in range(num_groups):
            start, end = g * num_generations, (g + 1) * num_generations
            group_log_w = log_w[start:end]
            group_w_norm = F.softmax(group_log_w, dim=0)
            ess = 1.0 / torch.sum(group_w_norm**2)

            if ess < num_generations * smc_ess_threshold: #(e.g. 50% of N)
                # --- WANDB LOGGING BLOCK ---
                resampling_step_counter += 1
                if smc_table is not None:
                    group_conf_tensor = torch.tensor(step_conf_values[start:end], device=input_ids.device)
                    group_avg_conf = avg_conf_values[start:end]
                    smc_table.add_data(
                        global_step,
                        resampling_step_counter,
                        g,
                        ess.item(),
                        group_conf_tensor.mean().item(),
                        group_conf_tensor.std().item(),
                        group_avg_conf.mean().item(),
                        group_avg_conf.std().item(),
                        group_w_norm.mean().item(),
                        group_w_norm.std().item()
                    )
                # --- END LOGGING BLOCK ---

                resampled_indices_local = ordered_stratified_resampling(group_w_norm.unsqueeze(0)).squeeze(0) #TODO: Srinivasan Sampling Process
                final_indices = resampled_indices_local + start
                
                input_ids[start:end] = input_ids[final_indices]
                unfinished_sequences[start:end] = unfinished_sequences[final_indices]
                if "attention_mask" in model_kwargs: 
                    model_kwargs["attention_mask"][start:end] = model_kwargs["attention_mask"][final_indices]
                if "past_key_values" in model_kwargs and model_kwargs["past_key_values"]:
                     for layer in model_kwargs["past_key_values"]:
                        for tensor_idx in range(len(layer)):
                            layer[tensor_idx][start:end] = layer[tensor_idx][final_indices]
                
                conf_deques[start:end] = [conf_deques[i] for i in final_indices.tolist()]
                conf_sums[start:end] = [conf_sums[i] for i in final_indices.tolist()]
                log_w[start:end] = 0.0
        
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
        output.smc_log_weights = log_w
        if confidences_tensor is not None:
            output.smc_confidences = confidences_tensor
        return output
    else:
        return input_ids