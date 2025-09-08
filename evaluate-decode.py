"""
Compare decoding strategies (default HF generate vs custom SMC generate)
on HF checkpoints without LoRA, across multiple datasets, with W&B logging.

Usage:
  uv run python3 evaluate-decode.py
  # Override config on CLI if needed, eg. change model or batch size
  uv run python3 evaluate-decode.py model.model_name=Qwen/Qwen2.5-3B
"""

import os
import json
from dataclasses import asdict

# Progress bar
try:
    from tqdm.auto import tqdm
except Exception:  # fallback if tqdm is unavailable
    def tqdm(*args, **kwargs):
        class _Noop:
            def __init__(self, total=None, **_):
                self.total = total
            def update(self, n=1):
                pass
            def close(self):
                pass
            def __enter__(self):
                return self
            def __exit__(self, exc_type, exc, tb):
                self.close()
        # If first arg is iterable, just return it
        if args and hasattr(args[0], "__iter__"):
            return args[0]
        return _Noop(total=kwargs.get("total"))

import hydra
from omegaconf import DictConfig, OmegaConf

import torch
from transformers import GenerationConfig

from unsloth import FastLanguageModel

from utils.data import (
    answer_correct,
    format_correct,
    get_questions,
    set_tokenizer_name,
)


def _ensure_pad_token(tokenizer):
    if tokenizer.pad_token is None:
        # Use eos as pad if pad doesn't exist
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})


def _build_gen_config_from_section(tokenizer, section_cfg: DictConfig, max_new_tokens: int) -> GenerationConfig:
    _ensure_pad_token(tokenizer)
    return GenerationConfig(
        do_sample=section_cfg.get("do_sample", False),
        temperature=section_cfg.get("temperature", 1.0),
        top_p=section_cfg.get("top_p", 1.0),
        top_k=section_cfg.get("top_k", 0),
        max_new_tokens=max_new_tokens,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        return_dict_in_generate=False,
    )


def _batch_tokenize(tokenizer, texts, max_length):
    return tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )


def _decode_generated(tokenizer, sequences: torch.Tensor, prompt_lens: torch.Tensor):
    # sequences: [B, T], prompt_lens: [B]
    texts = []
    pad_token_id = tokenizer.pad_token_id
    for i in range(sequences.size(0)):
        start = int(prompt_lens[i].item())
        gen_tokens = sequences[i, start:]
        # trim trailing pads for cleaner decoding
        if pad_token_id is not None:
            valid_len = (gen_tokens != pad_token_id).sum().item()
            if valid_len > 0:
                gen_tokens = gen_tokens[:valid_len]
        texts.append(tokenizer.decode(gen_tokens, skip_special_tokens=True))
    return texts


def _evaluate_once_default(model, tokenizer, prompts, answers, cfg: DictConfig, progress_desc: str | None = None):
    """Default decoding. If cfg.eval.default_use_vllm is True, use Unsloth+vLLM
    fast_generate for better memory/throughput (like evaluate-run.py). Otherwise
    use vanilla HF generate.
    """
    # Only use Unsloth+vLLM fast_generate when both flags are enabled.
    # Prevents passing vLLM-only kwargs to HF generate when fast_inference is off.
    use_vllm = bool(cfg.eval.get("default_use_vllm", True)) and bool(cfg.model.get("fast_inference", True))
    total_queries = len(prompts)
    ans_correct = 0
    for_correct = 0
    both_correct = 0
    total_gen_len = 0
    examples = []

    if use_vllm:
        from vllm import SamplingParams
        # Map HF-style knobs to vLLM SamplingParams
        top_k = cfg.default_decode.get("top_k", 0)
        vllm_top_k = int(top_k) if int(top_k) > 0 else -1
        sampling_params = SamplingParams(
            temperature=float(cfg.default_decode.get("temperature", 1.0)),
            top_p=float(cfg.default_decode.get("top_p", 1.0)),
            top_k=vllm_top_k,
            max_tokens=int(cfg.eval.max_new_tokens),
        )

        with tqdm(total=total_queries, desc=progress_desc or "default", unit="prompt", dynamic_ncols=True) as pbar:
            for i in range(0, total_queries, cfg.eval.batch_size):
                j = min(i + cfg.eval.batch_size, total_queries)
                batch_prompts = prompts[i:j]
                outputs = model.fast_generate(
                    batch_prompts,
                    sampling_params=sampling_params,
                    lora_request=None,
                    use_tqdm=False,
                )
                for k, output, gt in zip(range(i, j), outputs, answers[i:j]):
                    text = output.outputs[0].text
                    total_gen_len += len(output.outputs[0].token_ids)
                    a = answer_correct(text, gt)
                    f = format_correct(text)
                    ans_correct += a
                    for_correct += f
                    both_correct += (a and f)
                    if len(examples) < cfg.eval.sample_cnt:
                        examples.append({
                            "prompt": prompts[k],
                            "answer": gt,
                            "completion": text,
                            "correct": a,
                            "format_correct": f,
                        })
                pbar.update(j - i)
    else:
        # Vanilla HF generate path
        device = next(model.parameters()).device
        gen_cfg = _build_gen_config_from_section(tokenizer, cfg.default_decode, cfg.eval.max_new_tokens)
        with tqdm(total=total_queries, desc=progress_desc or "default", unit="prompt", dynamic_ncols=True) as pbar:
            for i in range(0, total_queries, cfg.eval.batch_size):
                j = min(i + cfg.eval.batch_size, total_queries)
                batch_prompts = prompts[i:j]
                enc = _batch_tokenize(tokenizer, batch_prompts, cfg.model.max_seq_length)
                input_ids = enc.input_ids.to(device)
                attn = enc.attention_mask.to(device)
                prompt_lens = attn.sum(dim=1)

                with torch.no_grad():
                    sequences = model.generate(
                        input_ids=input_ids,
                        attention_mask=attn,
                        generation_config=gen_cfg,
                    )

                texts = _decode_generated(tokenizer, sequences, prompt_lens)
                # Compute generated length per sample
                if tokenizer.pad_token_id is not None:
                    seq_valid_lens = (sequences != tokenizer.pad_token_id).sum(dim=1)
                else:
                    # If no pad, assume full length as valid
                    seq_valid_lens = torch.full((sequences.size(0),), sequences.size(1), device=sequences.device)
                gen_lens = (seq_valid_lens - prompt_lens).tolist()
                total_gen_len += sum(gen_lens)

                for k, text, gt in zip(range(i, j), texts, answers[i:j]):
                    a = answer_correct(text, gt)
                    f = format_correct(text)
                    ans_correct += a
                    for_correct += f
                    both_correct += (a and f)
                    # keep a few examples
                    if len(examples) < cfg.eval.sample_cnt:
                        examples.append({
                            "prompt": prompts[k],
                            "answer": gt,
                            "completion": text,
                            "correct": a,
                            "format_correct": f,
                        })
                pbar.update(j - i)

    avg_len = total_gen_len / total_queries if total_queries > 0 else 0.0
    return (
        ans_correct / total_queries,
        for_correct / total_queries,
        both_correct / total_queries,
        avg_len,
        examples,
    )


def _evaluate_once_custom(model, tokenizer, prompts, answers, cfg: DictConfig, logging_enabled: bool, progress_desc: str | None = None):
    from custom_generate.generate import generate as smc_generate

    device = next(model.parameters()).device

    # Build custom generation config with SMC knobs
    _ensure_pad_token(tokenizer)
    gen_cfg = GenerationConfig(
        do_sample=cfg.custom_decode.get("do_sample", True),
        temperature=cfg.custom_decode.get("temperature", 1.0),
        top_p=cfg.custom_decode.get("top_p", 1.0),
        top_k=cfg.custom_decode.get("top_k", 0),
        max_new_tokens=cfg.eval.max_new_tokens,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        return_dict_in_generate=False,
    )
    # SMC flags
    gen_cfg.use_smc = bool(cfg.custom_decode.get("use_smc", True))
    gen_cfg.num_generations = int(cfg.custom_decode.get("num_generations", 8))
    gen_cfg.smc_beta = float(cfg.custom_decode.get("smc_beta", 1.0))
    gen_cfg.smc_temperature = float(cfg.custom_decode.get("smc_temperature", 1.0))
    gen_cfg.smc_warmup_tokens = int(cfg.custom_decode.get("smc_warmup_tokens", 10))
    gen_cfg.smc_max_resampling_steps = int(cfg.custom_decode.get("smc_max_resampling_steps", 5))
    # Additional SMC controls (kept for parity with training config)
    gen_cfg.smc_confidence_eta = float(cfg.custom_decode.get("smc_confidence_eta", 1.0))
    gen_cfg.smc_ess_threshold = float(cfg.custom_decode.get("smc_ess_threshold", 0.2))
    gen_cfg.smc_confidence_window_size = int(cfg.custom_decode.get("smc_confidence_window_size", 50))

    # logging for custom generator
    logging_config = {
        "is_main_process": True,
        "report_to": ["wandb"] if logging_enabled else [],
        "global_step": 0,
    }

    total_queries = len(prompts)
    ans_correct = 0
    for_correct = 0
    both_correct = 0
    total_gen_len = 0
    examples = []

    G = cfg.eval.batch_size_groups
    N = gen_cfg.num_generations

    with tqdm(total=total_queries, desc=progress_desc or "custom", unit="prompt", dynamic_ncols=True) as pbar:
        for i in range(0, total_queries, G):
            j = min(i + G, total_queries)
            batch_prompts = prompts[i:j]
            enc = _batch_tokenize(tokenizer, batch_prompts, cfg.model.max_seq_length)
            input_ids = enc.input_ids.to(device)
            attn = enc.attention_mask.to(device)
            prompt_lens = attn.sum(dim=1)

            # repeat each prompt N times for SMC groups
            input_rep = input_ids.repeat_interleave(N, dim=0)
            attn_rep = attn.repeat_interleave(N, dim=0)

            with torch.no_grad():
                sequences = smc_generate(
                    model,
                    input_ids=input_rep,
                    attention_mask=attn_rep,
                    generation_config=gen_cfg,
                    tokenizer=tokenizer,
                    logging_config=logging_config,
                )

            # Choose the first particle per group as the representative sample
            B_rep, T = sequences.shape
            assert B_rep % N == 0, "Batch size must be a multiple of num_generations"
            groups = B_rep // N
            sequences = sequences.view(groups, N, T)[:, 0, :].contiguous()

            texts = _decode_generated(tokenizer, sequences, prompt_lens)

            # Compute generated length per sample
            if tokenizer.pad_token_id is not None:
                seq_valid_lens = (sequences != tokenizer.pad_token_id).sum(dim=1)
            else:
                seq_valid_lens = torch.full((sequences.size(0),), sequences.size(1), device=sequences.device)
            gen_lens = (seq_valid_lens - prompt_lens).tolist()
            total_gen_len += sum(gen_lens)

            for k, text, gt in zip(range(i, j), texts, answers[i:j]):
                a = answer_correct(text, gt)
                f = format_correct(text)
                ans_correct += a
                for_correct += f
                both_correct += (a and f)
                if len(examples) < cfg.eval.sample_cnt:
                    examples.append({
                        "prompt": prompts[k],
                        "answer": gt,
                        "completion": text,
                        "correct": a,
                        "format_correct": f,
                    })
            pbar.update(j - i)

    avg_len = total_gen_len / total_queries if total_queries > 0 else 0.0
    return (
        ans_correct / total_queries,
        for_correct / total_queries,
        both_correct / total_queries,
        avg_len,
        examples,
    )


def _evaluate_dataset(model, tokenizer, dataset_name: str, cfg: DictConfig, wandb_run):
    # Style selection matches training/eval convention
    style = "instruct" if cfg.model.model_name.endswith("Instruct") else "base"
    ds = get_questions(dataset_name, split=cfg.split, style=style)
    prompts = [ex["prompt"] for ex in ds]
    answers = ds["answer"]

    # Accumulate per-repeat results
    results = {}

    run_default = bool(cfg.eval.get("run_default", True))
    run_custom = bool(cfg.eval.get("run_custom", True))

    if run_default:
        default_ans, default_for, default_both, default_len, default_examples = [], [], [], [], []
    if run_custom:
        custom_ans, custom_for, custom_both, custom_len, custom_examples = [], [], [], [], []

    for t in range(cfg.eval.repeat_cnt):
        if run_default:
            da, df, db, dl, de = _evaluate_once_default(
                model, tokenizer, prompts, answers, cfg,
                progress_desc=f"{dataset_name} | default | rep {t+1}/{cfg.eval.repeat_cnt}"
            )
            default_ans.append(da); default_for.append(df); default_both.append(db); default_len.append(dl)
            default_examples = de if not default_examples else default_examples
        if run_custom:
            ca, cf, cb, cl, ce = _evaluate_once_custom(
                model, tokenizer, prompts, answers, cfg,
                logging_enabled=cfg.wandb.enable,
                progress_desc=f"{dataset_name} | custom | rep {t+1}/{cfg.eval.repeat_cnt}"
            )
            custom_ans.append(ca); custom_for.append(cf); custom_both.append(cb); custom_len.append(cl)
            custom_examples = ce if not custom_examples else custom_examples

        if wandb_run is not None:
            log_payload = {"repeat": t}
            if run_default:
                log_payload.update({
                    f"default/{dataset_name}/ans_acc": da,
                    f"default/{dataset_name}/for_acc": df,
                    f"default/{dataset_name}/both_acc": db,
                    f"default/{dataset_name}/avg_len": dl,
                })
            if run_custom:
                log_payload.update({
                    f"custom/{dataset_name}/ans_acc": ca,
                    f"custom/{dataset_name}/for_acc": cf,
                    f"custom/{dataset_name}/both_acc": cb,
                    f"custom/{dataset_name}/avg_len": cl,
                })
            wandb_run.log(log_payload)

    if run_default:
        results["default"] = {
            "ans_acc": default_ans,
            "for_acc": default_for,
            "both_acc": default_both,
            "lengths": default_len,
            "examples": default_examples,
        }
    if run_custom:
        results["custom"] = {
            "ans_acc": custom_ans,
            "for_acc": custom_for,
            "both_acc": custom_both,
            "lengths": custom_len,
            "examples": custom_examples,
        }
    return results


@hydra.main(version_base=None, config_path="config", config_name="decode_eval")
def main(cfg: DictConfig) -> None:
    # For dataset tokenizer-dependent filtering
    set_tokenizer_name(cfg.model.model_name)

    # Load model and tokenizer via Unsloth, but disable fast_inference for HF generate
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg.model.model_name,
        max_seq_length=cfg.model.max_seq_length,
        load_in_4bit=cfg.model.load_in_4bit,
        fast_inference=cfg.model.fast_inference,
        gpu_memory_utilization=cfg.model.gpu_memory_utilization,
    )

    # W&B init
    wandb_run = None
    if cfg.wandb.enable:
        import wandb
        wandb_run = wandb.init(
            project=cfg.wandb.project_name,
            name=cfg.wandb.run_name,
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    results = {}
    for dataset_name in cfg.datasets:
        print(f"Evaluating dataset: {dataset_name}")
        results[dataset_name] = _evaluate_dataset(model, tokenizer, dataset_name, cfg, wandb_run)

    # Persist results under results/decode_compare/<run_name>.json
    results_dir = os.path.join("results", "decode_compare")
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, f"{cfg.wandb.run_name}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {out_path}")

    if wandb_run is not None:
        # Log final per-dataset mean accuracies
        summary = {}
        for d, res in results.items():
            for mode in res.keys():
                summary[f"{mode}/{d}/ans_acc_mean"] = float(sum(res[mode]["ans_acc"]) / max(1, len(res[mode]["ans_acc"])))
                summary[f"{mode}/{d}/for_acc_mean"] = float(sum(res[mode]["for_acc"]) / max(1, len(res[mode]["for_acc"])))
                summary[f"{mode}/{d}/both_acc_mean"] = float(sum(res[mode]["both_acc"]) / max(1, len(res[mode]["both_acc"])))
                summary[f"{mode}/{d}/length_mean"] = float(sum(res[mode]["lengths"]) / max(1, len(res[mode]["lengths"])))
        wandb_run.log(summary)
        wandb_run.finish()


if __name__ == "__main__":
    main()
