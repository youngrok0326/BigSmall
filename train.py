"""
To train a reasoning model using the Unsloth framework.

"""

import hydra
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="config", config_name="train")
def main(cfg: DictConfig) -> None:
    from utils.patcher import apply_patch
    apply_patch(cfg.rl.algorithm)
    # Patch the trl trainers to use FastLanguageModel
    from unsloth import FastLanguageModel
    from transformers import AutoTokenizer
    
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
    dataset_training = get_questions(cfg.rl.dataset, split = "train", 
                                     style = "instruct" if cfg.model.model_name[-8:] == "Instruct" else "base")
    # dataset_testing  = get_math8k_questions(split = "test")

    # Import the trainer and config
    from trl import GRPOTrainer, GRPOConfig
    Trainer = GRPOTrainer
    Config = GRPOConfig
    
    # Initialize the trainer
    rlparams = {
        "algorithm": cfg.rl.algorithm,
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
        "loss_type": cfg.rl.loss_type,
        "mask_truncated_completions": cfg.rl.mask_truncated_completions
    }
    if "smc" in cfg.rl.algorithm.lower():
        smcparams = {
            "smc_temperature": cfg.smc.smc_temperature,
            "smc_warmup_tokens": cfg.smc.smc_warmup_tokens,
            "smc_max_resampling_steps": cfg.smc.smc_max_resampling_steps,
            "smc_step_delimiter_string": cfg.smc.smc_step_delimiter_string,
            "smc_beta": cfg.smc.smc_beta,
        } #TODO: check if this is correctly fed into a config after unsloth patching
    else:
        smcparams = {} 
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
        **rlparams,
        **smcparams
    )
    reward_funcs = [correctness_reward_func]
    if cfg.rl.dataset == "gsm8k":
        reward_funcs = reward_funcs + [format_reward_func, xmlcount_reward_func]
    trainer = Trainer(
        model = model,
        processing_class = tokenizer,
        reward_funcs = reward_funcs,
        args = training_args,
        train_dataset = dataset_training,
    )

    import time
    from transformers import TrainerCallback, TrainerControl
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
    trainer.train(resume_from_checkpoint=cfg.rl.resume_from_checkpoint)

if __name__ == "__main__":
    main()