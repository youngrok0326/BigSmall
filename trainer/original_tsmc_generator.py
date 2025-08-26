# Copyright 2024 The GPT-Accelera Team
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import csv
import gc
import os
import sys
from copy import deepcopy
import time
from pathlib import Path
from typing import Optional, Tuple, Dict
from collections import OrderedDict
import itertools
import fcntl
import numpy as np

import torch
from torch.distributed import _functional_collectives as funcol
from models.reward_model import RewardModel, apply_reward_modeling_head
from torch.nn.utils.rnn import pad_sequence
from models.tp import apply_reward_head_tp
import torch._inductor.config
import torch._dynamo.config
from math_utils.grader import grade_answer
torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.triton.unique_kernel_names = True
torch._inductor.config.fx_graph_cache = True  # Experimental feature to reduce compilation times, will be on by default in future

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from models.model import Transformer, KVCacheObject, find_multiple, configs
from models.tp import maybe_init_dist, initialize_model_parallel, apply_tp
from models.tp import (
    get_model_parallel_rank,
    get_model_parallel_group,
    get_data_parallel_rank,
    get_data_parallel_world_size,
)
from models.tokenizer_utils import (
    AcceleraTokenizer,
    batch_encode_tokens,
)
from checkpoint_utils import (
    get_latest_checkpoint_path,
    load_inference_checkpoint,
)


def ordered_stratified_resampling_multi(weights_matrix):
    """
    Performs stratified resampling for multiple distributions simultaneously.

    Parameters:
    weights_matrix (numpy array): 2D array of shape (n, m) where n is the number of distributions
                                  and m is the number of particles.

    Returns:
    numpy array: 2D array of shape (n, m) with the resampled indices for each distribution.
    """
    n, m = weights_matrix.shape
    # Step 1: Generate stratified positions for all distributions
    positions = (torch.arange(m, device=weights_matrix.device) + torch.rand_like(weights_matrix)) / m
    
    # Step 2: Order the weights in ascending order
    sorted_indices = torch.argsort(weights_matrix, dim=1)
    weights_matrix = torch.take_along_dim(weights_matrix, sorted_indices, dim=1)
    
    # Step 3: Compute the cumulative sum for each distribution
    cumulative_sum = torch.cumsum(weights_matrix, dim=1)
    # Step 4: Use searchsorted to find the indices in a vectorized way
    
    indices = torch.searchsorted(cumulative_sum, positions, side='right')
    resample_indices = torch.take_along_dim(sorted_indices, indices, dim=1)
    return resample_indices

def _load_reward_model(checkpoint_path, device, precision, use_tp):
    with torch.device("meta"):
        model = RewardModel.from_name(checkpoint_path.parent.name)

    if "int8" in str(checkpoint_path):
        raise NotImplementedError("int8 quantization cannot be used for reward model!")

    if "int4" in str(checkpoint_path):
        raise NotImplementedError("int4 quantization cannot be used for reward model!")

    checkpoint = torch.load(str(checkpoint_path), mmap=True, weights_only=True)
    model.backbone_model.load_state_dict(checkpoint, assign=True)

    if use_tp:
        print("Applying tensor parallel to model ...")
        apply_tp(model.backbone_model)

    # todo: remove this when finish debugging    
    apply_reward_modeling_head(model.backbone_model)

    if use_tp:
        print("Applying tensor parallel to reward head ...")
        apply_reward_head_tp(model.backbone_model)

    model = model.to(device=device, dtype=precision)
    return model.eval()

@torch.no_grad()
def model_score(
    model: RewardModel,
    prompt: torch.Tensor,
    max_seq_len: Optional[int] = None,
) -> torch.Tensor:
    """
    Scores a batch of prompts using a reward model.
    """
    B, T = prompt.size(0), prompt.size(1)

    max_seq_len = max_seq_len or T

    device = prompt.device
    with torch.device(device):
        model.backbone_model.setup_caches(
            max_batch_size=B, max_seq_length=max_seq_len, kv_cache=False
        )

    with torch.backends.cuda.sdp_kernel(
        enable_flash=False, enable_mem_efficient=False, enable_math=True
    ):
        rewards = model(prompt)

    return rewards

def multinomial_sample_one_no_sync(
    probs_sort,
):  # Does multinomial sampling without a cuda synchronization
    q = torch.empty_like(probs_sort).exponential_(1)
    return torch.argmax(probs_sort / q, dim=-1, keepdim=True).to(dtype=torch.int)
    # return torch.argmax(probs_sort, dim=-1, keepdim=True).to(dtype=torch.int)


def logits_to_probs(logits, temperature: float = 1.0, top_k: Optional[int] = None):
    logits = logits / max(temperature, 1e-5)

    if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)), dim=-1)
        pivot = v.select(-1, -1).view(-1, 1)
        logits = torch.where(logits < pivot, -float("Inf"), logits)
    probs = torch.nn.functional.softmax(logits, dim=-1)
    return probs


def sample(
    logits, vocab_parallel, temperature: float = 1.0, top_k: Optional[int] = None
):
    with torch.autocast(device_type="cuda", enabled=False):
        logits = logits[:, -1].float()

        if vocab_parallel:
            logits = funcol.all_gather_tensor(
                logits, gather_dim=-1, group=get_model_parallel_group()
            )

        probs = logits_to_probs(logits, temperature, top_k)
        idx_next = multinomial_sample_one_no_sync(probs)
    return idx_next, probs


def remove_all_backward_hooks(model: torch.nn.Module) -> Dict[str, OrderedDict]:
    all_backward_hooks = {}

    for name, module in model.named_modules():
        all_backward_hooks[name] = module._backward_hooks
        module._backward_hooks = OrderedDict()

    return all_backward_hooks


def prefill(
    model: Transformer,
    x: torch.Tensor,
    input_pos: torch.Tensor,
    left_pad_mask_pos: torch.Tensor,
    external_kv_cache=None,
    **sampling_kwargs,
) -> torch.Tensor:
    # input_pos: [B, S]
    logits = model(x, input_pos, left_pad_mask_pos, external_kv_cache=external_kv_cache)
    return sample(logits, model.vocab_parallel, **sampling_kwargs)


def decode_one_token(
    model: Transformer,
    x: torch.Tensor,
    input_pos: torch.Tensor,
    left_pad_mask_pos: torch.Tensor,
    external_kv_cache=None,
    **sampling_kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # input_pos: [B, 1]
    assert input_pos.shape[-1] == 1
    logits = model(x, input_pos, left_pad_mask_pos, external_kv_cache=external_kv_cache)
    return sample(logits, model.vocab_parallel, **sampling_kwargs)


def decode_n_tokens_step(
    model: Transformer,
    reward_model: Transformer,
    cur_token: torch.Tensor,
    cur_prompt: torch.Tensor,
    input_pos: torch.Tensor,
    left_pad_mask_pos: torch.Tensor,
    num_new_tokens: int,
    max_seq_length: int,
    eos_id: Optional[int] = None,
    external_kv_cache = None,
    tsmc_batch_size = 1,
    tsmc_temperature = 1.0,
    stop_steps = 0,
    warmup: Optional[int] = 0,
    **sampling_kwargs,
):

    eos_flag = None
    B, max_len = cur_prompt.shape
    assert B % tsmc_batch_size == 0
    weights = torch.zeros_like(cur_prompt)
    if eos_id is not None:
        eos_flag = torch.zeros(B, dtype=torch.bool, device=cur_token.device)
    T = input_pos.item()
    input_pos = input_pos.unsqueeze(-1).repeat(B, 1)
    score_flag = torch.zeros(B, dtype=torch.bool, device=cur_token.device)
    score_position = torch.zeros(B, dtype=torch.int, device=cur_token.device)+T
    log_value = torch.zeros(B, dtype=torch.float, device=cur_token.device)
    batch_idx = torch.arange(B).to(cur_token.device)
    last_decode_pos = score_position.clone() 
    log_rewards = torch.empty(B, 0, device=cur_token.device)
    log_values = torch.empty(B, 0, device=cur_token.device)
    i = 0
    num_steps = 0 
    
    # check the following
    # eos or limit must have score_flag True
    # we have the following cases for each seq
    # (1) in generation for the current step: score_flag==False, last_decode_pos==score_position 
    # Checklist: if the score position (\n\n, eos), update score flag and position, if eos or max num token, update eos flag
    # update cur_token, cur_prompt, input_pos, log_prob
    # (2) finish this step (after reaching \n\n, eos): score_flag==True, last_decode_pos!=score_position
    # Checklist: no update anymore
    # (3) has reached end since last step (eos no max num tokens): score_flag==True, last_decode_pos==score_position
    # Checklist: no update anymore
    # (4) has reached max num tokens this step: score_flag==True, last_decode_pos==score_position
    # Checklist: no update anymore
    
    
    while True:
        with torch.backends.cuda.sdp_kernel(
            enable_flash=False, enable_mem_efficient=False, enable_math=True
        ):  # Actually better for Inductor to codegen attention here
            indices = batch_idx[~score_flag]
            for kv_cache in external_kv_cache:
                kv_cache.set_indices(indices)
            next_token, next_prob = decode_one_token(
                model, cur_token[~score_flag], input_pos[~score_flag], left_pad_mask_pos[~score_flag] if left_pad_mask_pos is not None else None, external_kv_cache, **sampling_kwargs
            )

        #assert (score_position[last_eos_flag]+1 == input_pos[last_eos_flag].view(-1)).all()
        # update for seqs in generation
        input_pos[~score_flag] += 1
        #next_probs = torch.cat([next_probs, next_prob.clone().view(B, 1, -1)], dim=1) # does not really make sense here
        # ensure cur_token is always the token to be put in cur_prompt
        cur_token[~score_flag] = next_token.clone()
        cur_prompt[batch_idx[~score_flag], input_pos.view(-1)[~score_flag]] = next_token.view(-1).clone()
        #print(cur_prompt[batch_idx, input_pos.view(-1)]==cur_token.view(-1))
        assert (cur_prompt[batch_idx, input_pos.view(-1)]==cur_token.view(-1)).all()
        # do not update the probability distribution reaching the score position
        if eos_flag is not None:
            # only sentence in generation needs the update of the eos flag
            # type 1: reaching eos, also update the score position 
            # type 2: reaching max num token, do not update the score position 
            cur_eos_flag =  (cur_token == eos_id).view(-1) 
            score_position[cur_eos_flag] = input_pos[cur_eos_flag].view(-1)
            weights[batch_idx[cur_eos_flag], input_pos.view(-1)[cur_eos_flag]] = 1                

            limit_flag = (input_pos==T+num_new_tokens).view(-1) & (~cur_eos_flag)
            eos_flag = eos_flag | cur_eos_flag | limit_flag
            # update the score flag for completed generations
            score_flag[eos_flag] = True
        i += 1
        
        # update for the \n\n
        if i>=warmup:
            ind1 = cur_prompt[batch_idx, input_pos.view(-1)]==config['newline_token']
            ind2 = cur_prompt[batch_idx, input_pos.view(-1)-1]==config['newline_token']
            ind3 = cur_prompt[batch_idx, input_pos.view(-1)-2]==config['answer']
            ind4 = cur_prompt[batch_idx, input_pos.view(-1)-3]==config['#']
            newline_ind = (ind1 & ind2 & ~(ind3 & ind4)).view(-1)
            # only the sentence not reaching the score position would get updated
            # otherwise, the first score_position would be overwritten
            # no need to worry the limit flag since cur_prompt would not be updated in that case
            #newline_ind = newline_ind & (~score_flag) 
            # do not do resampling in the later stage to accelerate the convergence
            if num_steps < stop_steps :
                weights[batch_idx[newline_ind], input_pos.view(-1)[newline_ind]] = 1
                score_position = torch.where(newline_ind, input_pos.view(-1), score_position)
                # update score flag
                score_flag = score_flag | newline_ind 
        # at the end of each iteration, i should equal the number of new tokens generated of the shortest sequence

        if score_flag.all():
            num_steps += 1
            idle_flag = last_decode_pos==score_position
            if idle_flag.all():
                ## happens when all new generated seqs are unfinished due to max num tokens
                break

            y = model_score(
                reward_model,
                cur_prompt[:,:input_pos.max()+1],
                max_seq_len=max_seq_length,
            )

            log_weight = log_value.clone()
            log_score = torch.log(torch.sigmoid(y[batch_idx[~idle_flag], score_position[~idle_flag]])).float()*tsmc_temperature
            log_weight[~idle_flag] = log_score
            log_weight[limit_flag] = -1e6 # discard unfinished seqs
            log_ratio = log_weight - log_value
            log_value = log_weight.clone()
            log_rewards = torch.hstack([log_rewards, log_ratio.unsqueeze(1)])
            log_values = torch.hstack([log_values, log_value.unsqueeze(1)])
            if eos_flag is not None and eos_flag.all():
                break 

            log_ratio = log_ratio.reshape(-1, tsmc_batch_size)
            Z = torch.logsumexp(log_ratio, dim=1, keepdim=True)
            prob = torch.exp(log_ratio-Z)
            
            indices = ordered_stratified_resampling_multi(prob)

            no_change = torch.arange(B).reshape(-1, tsmc_batch_size).to(cur_prompt.device)
            group_eos_flag = eos_flag.reshape(-1, tsmc_batch_size).all(dim=1)
            indices += torch.arange(0,B,tsmc_batch_size).view(log_ratio.size(0),1).to(cur_prompt.device)

            indices = torch.where(group_eos_flag.unsqueeze(1).repeat(1,tsmc_batch_size), no_change, indices)
            indices = indices.flatten()

            # do the replacement after resampling
            cur_token = cur_token[indices].view(-1,1).clone()
            cur_prompt = cur_prompt[indices]
            weights = weights[indices]
            input_pos = input_pos[indices]
            assert (cur_token.view(-1)==cur_prompt[batch_idx,input_pos.view(-1)]).all()
            log_value = log_value[indices]
            eos_flag = eos_flag[indices]
            limit_flag = limit_flag[indices]
            score_position = score_position[indices]
            log_rewards = log_rewards[indices]
            log_values = log_values[indices]

            last_decode_pos = score_position.clone()

            for kv_cache in external_kv_cache:
                kv_cache.replace(indices)

            # score flag is always true when eos flag is true
            score_flag = eos_flag.clone()
            if not eos_flag.all():
                i = score_position[~eos_flag].min().item()-T

            #assert 13 in cur_prompt[:,i+T]
        if eos_flag is not None and eos_flag.all():
            break        

    # Final evaluation
    rewards = []
    values = []
    for i in range(B):
        num_steps = weights[i].sum().item()
        step_rewards = torch.exp(log_rewards[i][:num_steps]).tolist()
        step_values = torch.exp(log_values[i][:num_steps]).tolist()
        if limit_flag[i]:
            step_rewards.append(0)
            step_values.append(0)
        rewards.append(step_rewards)
        values.append(step_values)
    
    return cur_prompt, i, rewards, values



def model_forward(model, x, input_pos):
    return model(x, input_pos)


@torch.no_grad()
def generate(
    model: Transformer,
    reward_model: Transformer,
    prompt: torch.Tensor,
    left_pad_mask_pos: torch.Tensor,
    max_new_tokens: int,
    tokenizer,
    tsmc_temperature: Optional[float]=1.0,
    eos_id: Optional[int] = None,
    max_seq_len: Optional[int] = None,
    tsmc_batch_size: Optional[int] = 1,
    stop_steps: Optional[int] = 0,
    warmup: Optional[int]=0,
    **sampling_kwargs,
) -> torch.Tensor:
    """
    Takes a conditioning sequence (prompt) as input and continues to generate as many tokens as requested.
    """

    # create an empty tensor of the expected final shape and fill in the current tokens
    B = prompt.size(0)
    T = prompt.size(1)
    T_new = T + max_new_tokens
    #max_seq_length = min(T_new, model.config.block_size) + 1
    # max_seq_length = max_seq_len or model.config.block_size
    max_seq_length = model.config.block_size

    device, dtype = prompt.device, prompt.dtype
    with torch.device(device):
        model.setup_caches(max_batch_size=B, max_seq_length=max_seq_length, kv_cache=False)
    head_dim = model.config.dim // model.config.n_head
    # external kv cache to overcome the issue during torch.compile
    external_kv_cache = [KVCacheObject(
                    B, find_multiple(max_seq_length, 8), model.config.n_local_heads, head_dim, device=device
                ) for b in model.layers]

    if tokenizer.pad_id >= 0:
        pad_id = tokenizer.pad_id
    else:
        pad_id = tokenizer.unk_id 
    empty = torch.zeros((B, T_new), dtype=dtype, device=device) + pad_id
    empty[:, :T] = prompt
    seq = empty
    input_pos = torch.arange(0, T, device=device)

    with torch.backends.cuda.sdp_kernel(
        enable_flash=False, enable_mem_efficient=False, enable_math=True
    ):
        next_token, next_prob = prefill(
            model, prompt, input_pos, left_pad_mask_pos, external_kv_cache, **sampling_kwargs
        )

    seq[:, T] = next_token.view(B)
    input_pos = torch.tensor([T], device=device, dtype=torch.int)
    
    seq, num_decoded_tokens, rewards, values = decode_n_tokens_step(
        model,
        reward_model,
        next_token.view(B, -1),
        seq,
        input_pos,
        left_pad_mask_pos,
        max_new_tokens - 1,
        max_seq_length + 1,
        eos_id,
        external_kv_cache,
        tsmc_batch_size,
        tsmc_temperature,
        stop_steps,
        warmup,
        **sampling_kwargs,
    )

    return seq, rewards, values, num_decoded_tokens


def _load_model(checkpoint_path, device, precision, use_tp):
    with torch.device("meta"):
        model = Transformer.from_name(
            checkpoint_path.parent.name,
            freeze_tok_embeddings=True,
            freeze_output=True,
            freeze_norm=True,
            vocab_parallel=True,
        )

    if "int8" in str(checkpoint_path):
        print("Using int8 weight-only quantization!")
        from models.quantize import WeightOnlyInt8QuantHandler

        simple_quantizer = WeightOnlyInt8QuantHandler(model)
        model = simple_quantizer.convert_for_runtime()

    if "int4" in str(checkpoint_path):
        print("Using int4 quantization!")
        path_comps = checkpoint_path.name.split(".")
        assert path_comps[-2].startswith("g")
        groupsize = int(path_comps[-2][1:])
        from models.quantize import WeightOnlyInt4QuantHandler

        simple_quantizer = WeightOnlyInt4QuantHandler(model, groupsize)
        model = simple_quantizer.convert_for_runtime()

    checkpoint = torch.load(str(checkpoint_path), mmap=True, weights_only=True)
    model.load_state_dict(checkpoint, assign=True)

    if use_tp:
        print("Applying tensor parallel to model ...")
        apply_tp(model)

    model = model.to(device=device, dtype=precision)
    return model.eval()


def main(
    seed: int,
    prompt_file: Path,
    output_file: Path,
    batch_size: int = 4,
    tsmc_batch_size: int = 4,
    stop_steps: int = 3,
    num_samples: int = 5,
    max_new_tokens: int = 100,
    top_k: int = 200,
    temperature: float = 0.8,
    tsmc_temperature: float = 1.0,
    checkpoint_path: Path = Path(
        "checkpoints/meta-Transformer/Transformer-2-7b-chat-hf/model.pth"
    ),
    compile: bool = True,
    default_compile: bool = False,
    finetune_checkpoint_path: Optional[Path] = None,
    finetune_reward_checkpoint_path: Optional[Path] = None,
    finetune_checkpoint_prefix: Optional[str] = None,
    resume_generation: bool = False,
    tensor_parallel_size: Optional[int] = None,
    on_the_fly_8bit_quantization: bool = False,
    warmup: Optional[int]=0
) -> None:
    """Generates text samples based on a pre-trained Transformer model and tokenizer."""
    assert checkpoint_path.is_file(), checkpoint_path

    tokenizer_path = checkpoint_path.parent / "tokenizer.model"
    if not tokenizer_path.is_file():
        tokenizer_path = checkpoint_path.parent

    global config
    if 'llama' in str(checkpoint_path):
        config = configs['llama']
    elif 'llemma' in str(checkpoint_path):
        config = configs['llemma']
    elif 'deepseek' in str(checkpoint_path):
        config = configs['deepseek']
    else:
        raise NotImplementedError
    
    global print
    
    rank = maybe_init_dist()
    use_tp = rank is not None
    tp_size = 1
    if use_tp:
        tp_size = tensor_parallel_size or torch.distributed.get_world_size()
        initialize_model_parallel(tp_size)
        if rank != 0:
            # only print on rank 0
            print = lambda *args, **kwargs: None

    device = "cuda"
    precision = torch.bfloat16

    print("Loading model ...")
    t0 = time.time()
    model = _load_model(checkpoint_path, device, precision, use_tp)
    # switch back to the first line when finishing debugging
    reward_model = _load_reward_model(checkpoint_path, device, precision, use_tp)
    
    
    if finetune_checkpoint_path is not None:
        finetune_checkpoint_path, _, _ = get_latest_checkpoint_path(
            finetune_checkpoint_path,
            prefix=finetune_checkpoint_prefix,
        )

        if finetune_checkpoint_path is not None:
            print(f"Loading finetune model from {finetune_checkpoint_path} ...")
            load_inference_checkpoint(finetune_checkpoint_path, model)
        model = model.to(device=device)
        model = model.eval()
    
    if finetune_reward_checkpoint_path is not None:
        finetune_reward_checkpoint_path, _, _ = get_latest_checkpoint_path(
            finetune_reward_checkpoint_path,
            prefix=finetune_checkpoint_prefix,
        )

        if finetune_reward_checkpoint_path is not None:
            print(f"Loading finetune reward model from {finetune_reward_checkpoint_path} ...")
            load_inference_checkpoint(finetune_reward_checkpoint_path, reward_model)
        reward_model = reward_model.to(device=device)
        reward_model = reward_model.eval()        
    
    if on_the_fly_8bit_quantization:
        print("Quantizing model ...")
        from models.quantize import WeightOnlyInt8QuantHandler

        simple_quantizer = WeightOnlyInt8QuantHandler(model)
        model = simple_quantizer.convert_for_runtime_on_the_fly()
        model = model.to(device=device)
        model = model.eval()

        reward_model = simple_quantizer.convert_for_runtime_on_the_fly()
        reward_model = reward_model.to(device=device)
        reward_model = reward_model.eval()

    torch.cuda.synchronize()
    print(f"Time to load model: {time.time() - t0:.02f} seconds")

    tokenizer = AcceleraTokenizer(tokenizer_path)

    torch.manual_seed(seed)
    model_size = sum(
        [
            p.numel() * p.dtype.itemsize
            for p in itertools.chain(model.parameters(), model.buffers())
        ]
    )

    assert not (compile and default_compile), "Cannot compile with both modes"

    if compile or default_compile:
        global decode_one_token

    if compile:
        decode_one_token = torch.compile(
            decode_one_token, mode="reduce-overhead", fullgraph=True
        )

    if default_compile:
        decode_one_token = torch.compile(
            decode_one_token, mode="default", fullgraph=True
        )
    
    with open(prompt_file, "r") as f:
        prompts = json.load(f)

    # sort prompts by length to minimize padding

    prompt_idx = list(range(len(prompts)))

    assert "idx" not in prompts[0], "Prompts already have idx field"

    if "prompt" in prompts[0]:
        prompts = [
            {"idx": idx, "prompt": prompt["prompt"], 'gt_answer': prompt['gt_answer']}
            for idx, prompt in zip(prompt_idx, prompts)
        ]
    elif "input" in prompts[0]:
        prompts = [
            {"idx": idx, "prompt": prompt["input"], 'gt_answer': prompt['gt_answer']}
            for idx, prompt in zip(prompt_idx, prompts)
        ]
    else:
        raise ValueError("Prompts must have either prompt or input field")

    print("Tokenizing prompts ...")
    all_prompts = [prompt["prompt"] for prompt in prompts]
    
    tokenized_full_seq = tokenizer.batch_encode(
        all_prompts, bos=[False] * len(all_prompts), eos=[False] * len(all_prompts)
    )

    for prompt, tokenized in zip(prompts, tokenized_full_seq):
        prompt["prompt_len"] = len(tokenized)

    num_sample_prompts = []
    for prompt in prompts:
        for i in range(num_samples):
            sample_prompt = deepcopy(prompt)
            sample_prompt["sample_idx"] = i
            num_sample_prompts.append(sample_prompt)
    prompts = num_sample_prompts

    skipped_prompt_ids = dict()

    if rank == 0 or not use_tp:
        output_parent = output_file.parent
        if not output_parent.is_dir():
            output_parent.mkdir(exist_ok=True, parents=True)

    if use_tp:
        torch.distributed.barrier()

    if resume_generation and os.path.isfile(output_file):
        with open(output_file, "r") as f:
            for line in f:
                sample = json.loads(line)
                if sample["idx"] not in skipped_prompt_ids:
                    skipped_prompt_ids[sample["idx"]] = 0
                skipped_prompt_ids[sample["idx"]] += 1

    new_prompts = []
    for prompt in prompts:
        if prompt["idx"] not in skipped_prompt_ids:
            new_prompts.append(prompt)
        else:
            skipped_prompt_ids[prompt["idx"]] -= 1

            if skipped_prompt_ids[prompt["idx"]] == 0:
                del skipped_prompt_ids[prompt["idx"]]

    prompts = new_prompts

    while len(prompts) % batch_size != 0:
        prompts.insert(0, prompts[0])

    dp_rank = get_data_parallel_rank()
    tp_rank = get_model_parallel_rank()

    dp_size = get_data_parallel_world_size()

    if tp_rank == 0:
        output_writer = open(output_file, "a")

    batch_idx = 0

    gc.collect()
    torch.cuda.empty_cache()

    max_seq_len = prompts[-1]["prompt_len"] + max_new_tokens
    max_seq_len = min(max_seq_len, model.config.block_size)

    if compile:
        remove_all_backward_hooks(model)
        remove_all_backward_hooks(reward_model)
        global model_forward
        model_forward = torch.compile(
            model_forward, mode="reduce-overhead", fullgraph=True
        )

    for batched_prompt_idx in range(0, len(prompts), batch_size):
        batch_idx += 1
        if batch_idx % dp_size != dp_rank:
            continue

        batched_prompts = prompts[batched_prompt_idx : batched_prompt_idx + batch_size]

        encoded, left_pad_mask_pos = batch_encode_tokens(
            tokenizer, [_["prompt"] for _ in batched_prompts], bos=True, device=device
        )
        prompt_length = encoded.size(1)

        t0 = time.perf_counter()

        model_max_length = model.config.block_size
        local_max_new_tokens = max_new_tokens
        if local_max_new_tokens + prompt_length >= model_max_length:
            local_max_new_tokens = model_max_length - prompt_length - 1
        
        y, rewards, values, num_decoded_tokens = generate(
            model,
            reward_model,
            encoded,
            left_pad_mask_pos,
            local_max_new_tokens,
            tokenizer,
            temperature=temperature,
            tsmc_temperature=tsmc_temperature,
            top_k=top_k,
            eos_id=tokenizer.eos_id,
            max_seq_len=max_seq_len,
            tsmc_batch_size=tsmc_batch_size,
            stop_steps=stop_steps,
            warmup=warmup
        )

        full_y_list = y.tolist()
        print(post_process(full_y_list[0], tokenizer))
        print()

        t = time.perf_counter() - t0
        tokens_generated = num_decoded_tokens * y.size(0)
        tokens_sec = tokens_generated / t

        print(f"Prompt length: {prompt_length}")
        print(
            f"Time for inference {batched_prompt_idx + batch_size} / {len(prompts)}"
            f": {t:.02f} sec total, {tokens_sec:.02f} tokens/sec"
        )
        print(f"Bandwidth achieved: {model_size * tokens_sec / 1e9:.02f} GB/s")

        outputs = []

        for y_list in full_y_list:
            output = post_process(y_list[prompt_length:], tokenizer)
            outputs.append(output)

        if tp_rank == 0:
            fcntl.flock(output_writer, fcntl.LOCK_EX)
            try:
                for prompt, output, reward, value in zip(batched_prompts, outputs, rewards, values):
                    output_writer.write(
                        json.dumps(
                            {
                                "idx": prompt["idx"],
                                "sample_idx": prompt["sample_idx"],
                                "prompt": prompt["prompt"],
                                "output": output,
                                "reward": reward,
                                "value": value,
                            }
                        )
                        + "\n"
                    )
                output_writer.flush()
            finally:
                fcntl.flock(output_writer, fcntl.LOCK_UN)

def post_process(y_list, tokenizer):
    y_list = y_list[:]
    if tokenizer.eos_id in y_list:
        y_list = y_list[: y_list.index(tokenizer.eos_id)]

    if tokenizer.pad_id in y_list:
        y_list = y_list[::-1]
        y_list = y_list[: y_list.index(tokenizer.pad_id)]
        y_list = y_list[::-1]
    return tokenizer.decode(y_list)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Your CLI description.")

    parser.add_argument(
        "--seed", type=int, default=1234, help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--prompt_file",
        type=Path,
        required=True,
        help="File containing prompts, one per line.",
    )
    parser.add_argument(
        "--output_file",
        type=Path,
        required=True,
        help="File to write generated samples to.",
    )
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size.")
    parser.add_argument("--tsmc_batch_size", type=int, default=4, help="Batch size.")
    parser.add_argument("--stop_steps", type=int, default=5, help="Stop steps.")
    parser.add_argument("--num_samples", type=int, default=1, help="Number of samples.")
    parser.add_argument(
        "--max_new_tokens", type=int, default=200, help="Maximum number of new tokens."
    )
    parser.add_argument("--top_k", type=int, default=200, help="Top-k for sampling.")
    parser.add_argument(
        "--temperature", type=float, default=0.8, help="Temperature for sampling."
    )
    parser.add_argument(
        "--tsmc_temperature", type=float, default=1.0, help="Temperature for tsmc sampling."
    )
    parser.add_argument(
        "--checkpoint_path",
        type=Path,
        default=Path("checkpoints/meta-Transformer/Transformer-2-7b-chat-hf/model.pth"),
        help="Model checkpoint path.",
    )
    parser.add_argument(
        "--compile", action="store_true", help="Whether to compile the model."
    )
    parser.add_argument(
        "--default_compile",
        action="store_true",
        help="Whether to compile the model with default settings.",
    )
    parser.add_argument(
        "--finetune_checkpoint_path",
        type=Path,
        default=None,
        help="Finetune checkpoint path.",
    )
    parser.add_argument(
        "--finetune_reward_checkpoint_path",
        type=Path,
        default=None,
        help="Finetune reward checkpoint path.",
    )
    parser.add_argument(
        "--finetune_checkpoint_prefix",
        type=str,
        default=None,
        help="Finetune checkpoint prefix.",
    )
    parser.add_argument(
        "--resume_generation", action="store_true", help="Whether to resume generation."
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=None,
        help="Size of tensor parallelism.",
    )
    parser.add_argument(
        "--on_the_fly_8bit_quantization",
        action="store_true",
        help="Whether to quantize after loading the model.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=0,
        help="warmup token number before resampling",
    )
    
    args = parser.parse_args()
    main(
        args.seed,
        args.prompt_file,
        args.output_file,
        args.batch_size,
        args.tsmc_batch_size,
        args.stop_steps,
        args.num_samples,
        args.max_new_tokens,
        args.top_k,
        args.temperature,
        args.tsmc_temperature,
        args.checkpoint_path,
        args.compile,
        args.default_compile,
        args.finetune_checkpoint_path,
        args.finetune_reward_checkpoint_path,
        args.finetune_checkpoint_prefix,
        args.resume_generation,
        args.tensor_parallel_size,
        args.on_the_fly_8bit_quantization,
        args.warmup
    )
