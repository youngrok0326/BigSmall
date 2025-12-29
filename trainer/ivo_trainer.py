from typing import Optional, Union

import torch
from datasets import Dataset, IterableDataset
from transformers import PreTrainedModel, PreTrainedTokenizerBase, ProcessorMixin, TrainerCallback

from .grpo_trainer import GRPOTrainer
from .ivo_config import IVOConfig


class IVOTrainer(GRPOTrainer):
    _tag_names = ["trl", "ivo"]

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs,
        args: Optional[IVOConfig] = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]] = None,
        processing_class: Optional[Union[PreTrainedTokenizerBase, ProcessorMixin]] = None,
        reward_processing_classes: Optional[Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        peft_config: Optional["PeftConfig"] = None,
    ):
        if args is None:
            args = IVOConfig("ivo")
        super().__init__(
            model=model,
            reward_funcs=reward_funcs,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            reward_processing_classes=reward_processing_classes,
            callbacks=callbacks,
            optimizers=optimizers,
            peft_config=peft_config,
        )
        self.ivo_beta = getattr(args, "ivo_beta", 1.0)
        self.normalized_softlabel = getattr(args, "normalized_softlabel", True)
        if self.num_generations > 1:
            self._group_shuffle_size = self.num_generations

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The IVOTrainer does not support returning outputs")

        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)

        per_token_logps, _ = self._get_per_token_logps_and_entropies(
            model,
            input_ids,
            attention_mask,
            logits_to_keep,
            compute_entropy=False,
            pixel_values=inputs.get("pixel_values"),
            image_grid_thw=inputs.get("image_grid_thw"),
            pixel_attention_mask=inputs.get("pixel_attention_mask"),
            image_sizes=inputs.get("image_sizes"),
        )

        old_per_token_logps = inputs.get("old_per_token_logps")
        if old_per_token_logps is None:
            old_per_token_logps = per_token_logps.detach()

        scores = inputs.get("scores")
        if scores is None:
            raise ValueError("IVOTrainer requires `scores` in inputs.")
        scores = scores.to(per_token_logps.device)

        if self.normalized_softlabel:
            soft_label = torch.softmax(
                scores.view(-1, self.num_generations) / self.ivo_beta,
                dim=1,
            ).view(-1)
        else:
            soft_label = torch.exp(scores / self.ivo_beta)

        completion_mask = completion_mask.to(per_token_logps.dtype)
        token_diff = (per_token_logps - old_per_token_logps) * completion_mask
        diff_cum = token_diff.cumsum(dim=1)
        group_lse = diff_cum.view(-1, self.num_generations, diff_cum.shape[1]).logsumexp(dim=1)
        group_lse = group_lse.repeat_interleave(self.num_generations, dim=0)

        per_token_loss = -soft_label.unsqueeze(1) * (diff_cum - group_lse)
        token_counts = completion_mask.sum(dim=1).clamp(min=1.0)
        per_seq_loss = (per_token_loss * completion_mask).sum(dim=1) / token_counts
        loss = per_seq_loss.mean()

        if self.beta != 0.0:
            ref_per_token_logps = inputs.get("ref_per_token_logps")
            if ref_per_token_logps is None:
                raise ValueError("IVOTrainer requires `ref_per_token_logps` when beta != 0.0.")
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps)
                - (ref_per_token_logps - per_token_logps)
                - 1
            )
            mean_kl = (per_token_kl * completion_mask).sum() / completion_mask.sum().clamp(min=1.0)
            loss = loss + self.beta * mean_kl

            mode = "train" if self.model.training else "eval"
            self._metrics[mode]["kl"].append(self.accelerator.gather(mean_kl).mean().item())

        return loss
