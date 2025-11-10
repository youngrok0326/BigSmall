"""
To train a reasoning model using the Unsloth framework.

"""

import os

os.environ["VLLM_LOGGING_LEVEL"] = "WARNING"

import logging

logging.getLogger("vllm").setLevel(logging.WARNING)

import hydra
from omegaconf import DictConfig

from utils.logging_utils import setup_file_logging
from utils.geo_vllm import apply_geo_patch, restore_vllm
from utils.unsloth_patch import apply_unsloth_patch, restore_unsloth


@hydra.main(version_base=None, config_path="config", config_name="train")
def main(cfg: DictConfig) -> None:
    setup_file_logging("train.log")

    from utils.patcher import apply_patch
    apply_patch(cfg.rl.algorithm)
    if (cfg.rl.algorithm or "").lower() == "geogrpo":
        apply_geo_patch()
        apply_unsloth_patch()
    else:
        restore_vllm()
        restore_unsloth()
    from utils.data import set_tokenizer_name
    set_tokenizer_name(cfg.model.model_name)
    # Patch the trl trainers to use FastLanguageModel
    from unsloth import FastLanguageModel
    
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
    from trl import GRPOTrainer as Trainer, GRPOConfig as Config
    
    # Initialize the trainer
    rlparams = {
        "algorithm": cfg.rl.algorithm,
        "geo_lambda": cfg.rl.geo_lambda,
        "max_prompt_length": cfg.rl.max_prompt_length,
        "max_completion_length": cfg.rl.max_completion_length,
        "num_generations": cfg.rl.num_generations,
        "max_steps": cfg.rl.max_steps,
        "save_steps": cfg.rl.save_steps,
        "beta": cfg.rl.beta,
        "save_strategy": "steps" if cfg.rl.save_strategy == "steps" else "no",
        "epsilon": cfg.rl.epsilon,
        "epsilon_high": cfg.rl.epsilon_high,
        "delta": cfg.rl.delta,
        "scale_rewards": cfg.rl.scale_rewards,
        "loss_type": cfg.rl.loss_type,
        "mask_truncated_completions": cfg.rl.mask_truncated_completions
    }
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
    gen_kwargs: dict[str, object] = {}
    if (cfg.rl.algorithm or "").lower() == "geogrpo":
        gen_kwargs["geo_lambda"] = cfg.rl.geo_lambda
    generation_kwargs = gen_kwargs or None

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
