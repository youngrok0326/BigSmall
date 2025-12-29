import inspect
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
        self.teacher_model_id = getattr(args, "teacher_model", None)
        self.teacher_beta = float(getattr(args, "teacher_beta", 0.0) or 0.0)
        self.teacher_device = getattr(args, "teacher_device", None)
        self._teacher = None
        self._teacher_model_kwarg_keys = None
        self._teacher_device = None
        if self.teacher_beta > 0.0:
            if not self.teacher_model_id:
                raise ValueError("teacher_model must be set when teacher_beta > 0.")
            self._init_teacher()
        if self.num_generations > 1:
            self._group_shuffle_size = self.num_generations

    def _get_student_tokenizer(self):
        if isinstance(self.processing_class, ProcessorMixin):
            return self.processing_class.tokenizer
        return self.processing_class

    def _ensure_tokenizer_compatibility(self, teacher_tokenizer):
        student_tokenizer = self._get_student_tokenizer()
        if teacher_tokenizer is None or student_tokenizer is None:
            return
        if getattr(teacher_tokenizer, "vocab_size", None) != getattr(student_tokenizer, "vocab_size", None):
            raise ValueError("Teacher tokenizer vocab_size does not match the student tokenizer.")
        if getattr(teacher_tokenizer, "special_tokens_map", None) != getattr(student_tokenizer, "special_tokens_map", None):
            raise ValueError("Teacher tokenizer special tokens do not match the student tokenizer.")
        test_text = "Tokenization compatibility check."
        teacher_ids = teacher_tokenizer.encode(test_text, add_special_tokens=False)
        student_ids = student_tokenizer.encode(test_text, add_special_tokens=False)
        if teacher_ids != student_ids:
            raise ValueError("Teacher tokenizer encoding differs from the student tokenizer.")

    def _init_teacher(self):
        from unsloth import FastLanguageModel

        max_seq_length = getattr(self.model, "max_seq_length", None)
        if max_seq_length is None:
            max_seq_length = getattr(self.model.config, "max_position_embeddings", None)
        if max_seq_length is None:
            max_seq_length = 2048

        load_in_4bit = bool(getattr(self.model, "is_loaded_in_4bit", False))
        load_in_8bit = bool(getattr(self.model, "is_loaded_in_8bit", False))
        device_map = self.teacher_device or "sequential"

        teacher_model, teacher_tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.teacher_model_id,
            max_seq_length=max_seq_length,
            load_in_4bit=load_in_4bit,
            load_in_8bit=load_in_8bit,
            fast_inference=False,
            device_map=device_map,
        )
        self._ensure_tokenizer_compatibility(teacher_tokenizer)
        teacher_model.eval()
        if hasattr(teacher_model, "for_inference"):
            teacher_model.for_inference()
        for param in teacher_model.parameters():
            param.requires_grad_(False)
        self._teacher = teacher_model
        self._teacher_device = next(teacher_model.parameters()).device

        base_forward = teacher_model.forward
        if hasattr(teacher_model, "get_base_model"):
            base_forward = teacher_model.get_base_model().forward
        self._teacher_model_kwarg_keys = inspect.signature(base_forward).parameters.keys()

    def _get_teacher_values(
        self,
        input_ids,
        attention_mask,
        logits_to_keep,
        pixel_values=None,
        image_grid_thw=None,
        pixel_attention_mask=None,
        image_sizes=None,
    ):
        if self._teacher is None:
            raise ValueError("Teacher model is not initialized.")
        teacher = self._teacher
        teacher_device = self._teacher_device or input_ids.device
        batch_size = input_ids.size(0)
        all_values = []
        for start in range(0, input_ids.size(0), batch_size):
            input_ids_batch = input_ids[start : start + batch_size].to(teacher_device)
            attention_mask_batch = attention_mask[start : start + batch_size].to(teacher_device)

            model_inputs = {"input_ids": input_ids_batch, "attention_mask": attention_mask_batch}
            if image_grid_thw is not None and pixel_values is not None:
                model_inputs["image_grid_thw"] = image_grid_thw[start : start + batch_size].to(teacher_device)
                start_pixel_idx = image_grid_thw[:start].prod(-1).sum().item()
                end_pixel_idx = image_grid_thw[: start + batch_size].prod(-1).sum().item()
                model_inputs["pixel_values"] = pixel_values[start_pixel_idx:end_pixel_idx].to(teacher_device)
            elif pixel_values is not None:
                model_inputs["pixel_values"] = pixel_values[start : start + batch_size].to(teacher_device)
            if pixel_attention_mask is not None:
                model_inputs["pixel_attention_mask"] = pixel_attention_mask[start : start + batch_size].to(teacher_device)
            if image_sizes is not None:
                model_inputs["image_sizes"] = image_sizes[start : start + batch_size].to(teacher_device)
            if "logits_to_keep" in self._teacher_model_kwarg_keys:
                model_inputs["logits_to_keep"] = logits_to_keep + 1

            logits = teacher(**model_inputs).logits
            logits = logits[:, :-1, :]
            logits = logits[:, -logits_to_keep:, :]
            logits = logits / self.ivo_beta
            values = self.ivo_beta * torch.logsumexp(logits, dim=-1)
            all_values.append(values)

        values = torch.cat(all_values, dim=0)
        return values.to(input_ids.device)

    def _generate_and_score_completions(self, inputs):
        outputs = super()._generate_and_score_completions(inputs)
        if self._teacher is None or self.teacher_beta <= 0.0:
            return outputs

        prompt_ids, prompt_mask = outputs["prompt_ids"], outputs["prompt_mask"]
        completion_ids, completion_mask = outputs["completion_ids"], outputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)

        with torch.inference_mode():
            teacher_values = self._get_teacher_values(
                input_ids,
                attention_mask,
                logits_to_keep,
                pixel_values=outputs.get("pixel_values"),
                image_grid_thw=outputs.get("image_grid_thw"),
                pixel_attention_mask=outputs.get("pixel_attention_mask"),
                image_sizes=outputs.get("image_sizes"),
            )
        outputs["teacher_psi"] = teacher_values - teacher_values[:, :1]
        return outputs

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

        teacher_psi = inputs.get("teacher_psi")
        if self.teacher_beta > 0.0:
            if teacher_psi is None:
                raise ValueError("IVOTrainer requires `teacher_psi` when teacher_beta > 0.")
            teacher_psi = teacher_psi.to(diff_cum.dtype)
            diff_cum = diff_cum - (self.teacher_beta / self.ivo_beta) * teacher_psi
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
