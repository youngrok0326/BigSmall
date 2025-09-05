# custom_generate/generate.py

import torch
import torch.nn.functional as F
from typing import Any, Optional, Union, Callable

from transformers import (
    GenerationConfig,
    LogitsProcessorList,
    StoppingCriteriaList,
    AutoModelForCausalLM, # For type hinting
)
from transformers.generation.utils import (
    GenerateDecoderOnlyOutput,
    GenerateEncoderDecoderOutput,
)

def generate(
    model: AutoModelForCausalLM,
    inputs: Optional[torch.Tensor] = None,
    generation_config: Optional[GenerationConfig] = None,
    logits_processor: Optional[LogitsProcessorList] = None,
    stopping_criteria: Optional[StoppingCriteriaList] = None,
    prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], list[int]]] = None,
    synced_gpus: Optional[bool] = False,
    streamer: Optional[Any] = None,
    **kwargs,
) -> Union[GenerateDecoderOnlyOutput, GenerateEncoderDecoderOutput, torch.LongTensor]:
    """
    A custom generate function that faithfully replicates the standard sampling logic by calling
    the model's internal helper methods for setup and execution.
    """
    # 1. Prepare all configurations and inputs using the model's internal helpers.
    # This is the most robust way to ensure full feature compatibility.
    generation_config, model_kwargs = model._prepare_generation_config(generation_config, **kwargs)
    inputs_tensor, model_input_name, model_kwargs = model._prepare_model_inputs(
        inputs, generation_config.bos_token_id, model_kwargs
    )
    batch_size = inputs_tensor.shape[0]
    model._prepare_special_tokens(generation_config)

    # 2. Determine generation mode (we only support sampling/greedy here)
    is_greedy_gen_mode = (
        (generation_config.num_beams == 1)
        and (generation_config.num_beam_groups == 1)
        and not generation_config.do_sample
    )
    is_sample_gen_mode = (
        (generation_config.num_beams == 1)
        and (generation_config.num_beam_groups == 1)
        and generation_config.do_sample
    )
    if not is_greedy_gen_mode and not is_sample_gen_mode:
        raise ValueError(
            "This custom generate function only supports greedy search and multinomial sampling."
        )

    # 3. Prepare processors, criteria, and other parameters by calling the model's helpers
    logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
    stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()

    logits_processor = model._get_logits_processor(
        generation_config=generation_config,
        input_ids_seq_length=inputs_tensor.shape[-1],
        encoder_input_ids=inputs_tensor,
        prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
        logits_processor=logits_processor,
    )
    stopping_criteria = model._get_stopping_criteria(
        generation_config=generation_config, stopping_criteria=stopping_criteria
    )
    
    if model.config.is_encoder_decoder:
        input_ids, model_kwargs = model._prepare_decoder_input_ids_for_generation(
            batch_size=batch_size,
            model_input_name=model_input_name,
            model_kwargs=model_kwargs,
            decoder_start_token_id=generation_config.decoder_start_token_id,
            device=inputs_tensor.device,
        )
    else:
        input_ids = inputs_tensor

    # 4. Expand inputs for `num_return_sequences` using the model's static method
    input_ids, model_kwargs = model._expand_inputs_for_generation(
        input_ids=input_ids,
        expand_size=generation_config.num_return_sequences,
        is_encoder_decoder=model.config.is_encoder_decoder,
        **model_kwargs,
    )

    # 5. The main sampling logic, replicated from `_sample`
    # init values
    pad_token_id = generation_config.pad_token_id
    eos_token_id = generation_config.eos_token_id
    output_scores = generation_config.output_scores
    output_attentions = generation_config.output_attentions
    output_hidden_states = generation_config.output_hidden_states
    output_logits = generation_config.output_logits
    return_dict_in_generate = generation_config.return_dict_in_generate
    do_sample = generation_config.do_sample

    # init attention / hidden states / scores tuples
    scores = () if (return_dict_in_generate and output_scores) else None
    raw_logits = () if (return_dict_in_generate and output_logits) else None
    decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
    cross_attentions = () if (return_dict_in_generate and output_attentions) else None
    decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

    if return_dict_in_generate and model.config.is_encoder_decoder:
        encoder_attentions = model_kwargs["encoder_outputs"].get("attentions")
        encoder_hidden_states = model_kwargs["encoder_outputs"].get("hidden_states")

    # keep track of which sequences are already finished
    batch_size, cur_len = input_ids.shape
    this_peer_finished = False
    unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)

    # main loop using the model's internal helper for robust multi-GPU handling
    while model._has_unfinished_sequences(this_peer_finished, synced_gpus):
        # prepare model inputs
        model_inputs = model.prepare_inputs_for_generation(input_ids, **model_kwargs)

        # forward pass
        outputs = model(**model_inputs, return_dict=True)
        
        # update model_kwargs for the next step
        model_kwargs = model._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder=model.config.is_encoder_decoder
        )

        # if this GPU is finished, skip the rest of the loop
        if synced_gpus and this_peer_finished:
            cur_len += 1
            continue
        
        next_token_logits = outputs.logits[:, -1, :]
        next_token_scores = logits_processor(input_ids, next_token_logits)

        # Store scores, attentions, etc.
        if return_dict_in_generate:
            if output_scores: scores += (next_token_scores,)
            if output_logits: raw_logits += (next_token_logits,)
            if output_attentions:
                decoder_attentions += ((outputs.decoder_attentions,) if model.config.is_encoder_decoder else (outputs.attentions,))
                if model.config.is_encoder_decoder: cross_attentions += (outputs.cross_attentions,)
            if output_hidden_states:
                decoder_hidden_states += ((outputs.decoder_hidden_states,) if model.config.is_encoder_decoder else (outputs.hidden_states,))

        # select next token
        next_tokens = torch.argmax(next_token_scores, dim=-1) if not do_sample else torch.multinomial(F.softmax(next_token_scores, dim=-1), num_samples=1).squeeze(1)

        # finished sentences should have their next token be a padding token
        if eos_token_id is not None:
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

        # update generated ids and streamer
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        if streamer: streamer.put(next_tokens.cpu())

        # update stopping criteria
        unfinished_sequences &= ~stopping_criteria(input_ids, scores)
        this_peer_finished = unfinished_sequences.max() == 0
        cur_len += 1

    if streamer: streamer.end()

    # 6. Return results
    if return_dict_in_generate:
        if model.config.is_encoder_decoder:
            return GenerateEncoderDecoderOutput(sequences=input_ids, scores=scores, logits=raw_logits, encoder_attentions=encoder_attentions, encoder_hidden_states=encoder_hidden_states, decoder_attentions=decoder_attentions, cross_attentions=cross_attentions, decoder_hidden_states=decoder_hidden_states, past_key_values=model_kwargs.get("past_key_values"))
        else:
            return GenerateDecoderOnlyOutput(sequences=input_ids, scores=scores, logits=raw_logits, attentions=decoder_attentions, hidden_states=decoder_hidden_states, past_key_values=model_kwargs.get("past_key_values"))
    else:
        return input_ids