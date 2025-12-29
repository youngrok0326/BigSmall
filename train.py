"""
To train a reasoning model using the Unsloth framework.

"""

import os

os.environ["VLLM_LOGGING_LEVEL"] = "WARNING"

import logging

logging.getLogger("vllm").setLevel(logging.WARNING)

import hydra
from omegaconf import DictConfig, OmegaConf

from utils.logging_utils import setup_file_logging


@hydra.main(version_base=None, config_path="config/train", config_name="train")
def main(cfg: DictConfig) -> None:
    setup_file_logging("train.log")

    from utils.patcher import apply_patch
    apply_patch()
    # Ensure Unsloth recompiles trainers after patching TRL.
    import hashlib
    base_dir = os.environ.get("HYDRA_ORIGINAL_CWD", os.getcwd())
    patch_files = [
        os.path.join(base_dir, "trainer", "grpo_trainer.py"),
        os.path.join(base_dir, "trainer", "grpo_config.py"),
        os.path.join(base_dir, "trainer", "ivo_trainer.py"),
        os.path.join(base_dir, "trainer", "ivo_config.py"),
    ]
    hasher = hashlib.md5()
    for path in patch_files:
        try:
            with open(path, "rb") as f:
                hasher.update(f.read())
        except FileNotFoundError:
            continue
    compile_hash = hasher.hexdigest()[:8]
    compile_dir = os.path.join(base_dir, "unsloth_compiled_cache", f"trl_patch_{compile_hash}")
    os.environ.setdefault("UNSLOTH_COMPILE_LOCATION", compile_dir)
    os.environ.setdefault("UNSLOTH_COMPILE_OVERWRITE", "1")
    # Import Unsloth before anything that pulls in transformers.
    import unsloth
    from unsloth import FastLanguageModel
    from utils.data import set_tokenizer_name
    set_tokenizer_name(cfg.model.model_name)
    
    # Load the model
    pretrained_args = {
        "model_name": cfg.model.model_name,
        "max_lora_rank": cfg.model.lora_rank,
        "load_in_4bit": cfg.model.load_in_4bit,
        "fast_inference": cfg.model.fast_inference,
        "max_seq_length": cfg.model.max_seq_length,
        "gpu_memory_utilization": cfg.model.gpu_memory_utilization,
    }
    model, tokenizer = FastLanguageModel.from_pretrained(**pretrained_args)
    peft_args = {
        "r": cfg.model.lora_rank,
        "lora_alpha": cfg.model.lora_rank,
        "target_modules": cfg.model.target_modules,
        "use_gradient_checkpointing": "unsloth",
        "random_state": cfg.model.random_state,
    }
    model = FastLanguageModel.get_peft_model(model, **peft_args)

    # Load the datasets
    from utils.data import get_questions
    dataset_training = get_questions(
        cfg.rl.dataset,
        split="train",
        style="instruct" if cfg.model.model_name[-8:] == "Instruct" else "base",
    )
    # dataset_testing  = get_math8k_questions(split = "test")

    # Import the trainer and config
    algorithm = cfg.rl.algorithm.lower()
    if algorithm == "grpo":
        from trl.trainer.grpo_trainer import GRPOTrainer as Trainer
        from trl.trainer.grpo_config import GRPOConfig as Config
    elif algorithm in ("ivo", "ivo_distill"):
        from trl.trainer.ivo_trainer import IVOTrainer as Trainer
        from trl.trainer.ivo_config import IVOConfig as Config
    else:
        raise ValueError(f"Unknown algorithm: {cfg.rl.algorithm}")
    
    # Initialize the trainer
    rlparams = OmegaConf.to_container(cfg.rl, resolve=True) or {}
    if not isinstance(rlparams, dict):
        raise ValueError("cfg.rl must be a mapping of RL parameters")
    if "save_strategy" in rlparams:
        rlparams["save_strategy"] = (
            "steps" if rlparams["save_strategy"] == "steps" else "no"
        )
    rlparams = {k: v for k, v in rlparams.items() if v is not None}
    if algorithm != "ivo_distill":
        rlparams.pop("teacher_model", None)
        rlparams.pop("teacher_beta", None)
        rlparams.pop("teacher_device", None)
    else:
        if not rlparams.get("teacher_model"):
            raise ValueError("rl.teacher_model must be set when rl.algorithm=IVO_DISTILL.")
        if float(rlparams.get("teacher_beta", 0.0)) <= 0.0:
            raise ValueError("rl.teacher_beta must be > 0 when rl.algorithm=IVO_DISTILL.")
    if cfg.wandb.enable:
        import wandb
        wandb_run = wandb.init(
            project = cfg.wandb.project_name,
            name = cfg.wandb.run_name,
            config = rlparams,
        )
        
    from unsloth import is_bfloat16_supported
    from utils.data import correctness_reward_func, xmlcount_reward_func, format_reward_func
    from inspect import signature
    valid_params = signature(Config.__init__).parameters.keys()
    rlparams = {k: v for k, v in rlparams.items() if k in valid_params}
    generation_kwargs = None

    training_args = Config(
        learning_rate = cfg.optim.learning_rate,
        adam_beta1 = cfg.optim.adam_beta1,
        adam_beta2 = cfg.optim.adam_beta2,
        weight_decay = cfg.optim.weight_decay,
        warmup_ratio = cfg.optim.warmup_ratio,
        lr_scheduler_type = cfg.optim.lr_scheduler_type,
        optim = cfg.optim.optim,
        logging_steps = cfg.optim.logging_steps,
        bf16 = is_bfloat16_supported(),
        fp16 = not is_bfloat16_supported(),
        per_device_train_batch_size = cfg.optim.per_device_train_batch_size,
        gradient_accumulation_steps = cfg.optim.gradient_accumulation_steps,
        temperature = cfg.optim.temperature,
        max_grad_norm = cfg.optim.max_grad_norm,
        report_to = "wandb" if cfg.wandb.enable else None,
        output_dir = f"checkpoints/{cfg.wandb.run_name}",
        generation_kwargs=generation_kwargs,
        **rlparams,
    )
    reward_funcs = [correctness_reward_func, format_reward_func, xmlcount_reward_func]
    trainer = Trainer(
        model = model,
        processing_class = tokenizer,
        reward_funcs = reward_funcs,
        args = training_args,
        train_dataset = dataset_training,
    )

    import time
    from transformers import TrainerCallback, TrainerControl

    class FinalCheckpointCallback(TrainerCallback):
        """Force a checkpoint save when training finishes."""

        def __init__(self, trainer):
            self.trainer = trainer

        def on_train_end(self, args, state, control, **kwargs):
            if self.trainer.args.should_save:
                self.trainer._save_checkpoint(self.trainer.model, trial=None)
            return control

    class TimedCheckpointCallback(TrainerCallback):
        """
        Ask the Trainer to do a normal checkpoint whenever
        interval seconds have passed. This assumes one step
        takes at most interval seconds.
        """
        def __init__(self, interval, max_time):   # 30 min = 1800 s
            self.interval = interval
            self.start_time = None
            self.next_save = interval
            self.max_time = max_time

        def on_train_begin(self, args, state, control, **kwargs):
            self.start_time = time.time()

        def on_step_end(self, args, state, control: TrainerControl, **kwargs):
            curr_time = time.time() - self.start_time
            if curr_time >= self.next_save:
                control.should_save = True
                self.next_save += self.interval
                assert curr_time < self.next_save, "One step took too long!"
                print(f"Checkpoint saved at {curr_time:.2f} seconds")
                if curr_time >= self.max_time:
                    control.should_training_stop = True

    # Start training
    if cfg.rl.save_strategy == "time":
        trainer.add_callback(TimedCheckpointCallback(cfg.rl.save_interval, cfg.rl.max_time))
        trainer.add_callback(FinalCheckpointCallback(trainer))
    trainer.train(resume_from_checkpoint=cfg.rl.resume_from_checkpoint)

if __name__ == "__main__":
    main()
