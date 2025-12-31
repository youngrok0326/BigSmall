import inspect
import os
from typing import Optional, Union

import torch
from accelerate.utils import is_peft_model
from datasets import Dataset, IterableDataset
from transformers import PreTrainedModel, PreTrainedTokenizerBase, ProcessorMixin, TrainerCallback
from trl.trainer.utils import entropy_from_logits, selective_log_softmax

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
        if hasattr(model, "vllm_engine") and hasattr(args, "use_vllm"):
            if not getattr(args, "use_vllm", False):
                args.use_vllm = True
            args.vllm_mode = "colocate"
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
        self.teacher_lora_path = getattr(args, "teacher_lora_path", None)
        self._teacher = None
        self._teacher_model_kwarg_keys = None
        self._teacher_device = None
        self._eos_token_ids = None
        if self.teacher_beta > 0.0:
            if not self.teacher_model_id:
                raise ValueError("teacher_model must be set when teacher_beta > 0.")
            self._init_teacher()
        if self.num_generations > 1:
            self._group_shuffle_size = self.num_generations
        self._ensure_vllm_generate_lora_request()

    def _resolve_teacher_lora_path(self, path: Optional[str]) -> Optional[str]:
        if not path:
            return None
        if isinstance(path, str) and path.lower() in ("none", "null"):
            return None
        path = os.path.expanduser(path)
        if os.path.isabs(path):
            return path
        from hydra.utils import get_original_cwd
        base_dir = get_original_cwd()
        candidate = os.path.join(base_dir, path)
        return candidate if os.path.exists(candidate) else path

    def _ensure_vllm_generate_lora_request(self):
        if getattr(self, "_vllm_generate_wrapped", False):
            return
        if not getattr(self, "use_vllm", False) or getattr(self, "vllm_mode", None) != "colocate":
            return
        llm = getattr(self, "llm", None)
        if llm is None or not hasattr(llm, "generate"):
            return
        if not (is_peft_model(self.model) and hasattr(self.model, "load_lora")):
            return

        original_generate = llm.generate

        def generate_with_lora(*args, **kwargs):
            if "lora_request" not in kwargs:
                visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0").replace(",", "")
                lora_name = f"ivo_trainer_lora_model_{visible}"
                kwargs["lora_request"] = self.model.load_lora(lora_name, load_tensors=True)
            return original_generate(*args, **kwargs)

        llm.generate = generate_with_lora
        self._vllm_generate_wrapped = True

    def _get_per_token_logps_and_entropies(
        self,
        model,
        input_ids,
        attention_mask,
        logits_to_keep,
        batch_size=None,
        compute_entropy=False,
        pixel_values=None,
        image_grid_thw=None,
        pixel_attention_mask=None,
        image_sizes=None,
    ):
        batch_size = batch_size or input_ids.size(0)
        all_logps = []
        all_entropies = []
        for start in range(0, input_ids.size(0), batch_size):
            input_ids_batch = input_ids[start : start + batch_size]
            attention_mask_batch = attention_mask[start : start + batch_size]

            model_inputs = {"input_ids": input_ids_batch, "attention_mask": attention_mask_batch}

            if image_grid_thw is not None and pixel_values is not None:
                model_inputs["image_grid_thw"] = image_grid_thw[start : start + batch_size]
                start_pixel_idx = image_grid_thw[:start].prod(-1).sum().item()
                end_pixel_idx = image_grid_thw[: start + batch_size].prod(-1).sum().item()
                model_inputs["pixel_values"] = pixel_values[start_pixel_idx:end_pixel_idx]
            elif pixel_values is not None:
                model_inputs["pixel_values"] = pixel_values[start : start + batch_size]
            if pixel_attention_mask is not None:
                model_inputs["pixel_attention_mask"] = pixel_attention_mask[start : start + batch_size]
            if image_sizes is not None:
                model_inputs["image_sizes"] = image_sizes[start : start + batch_size]

            if "logits_to_keep" in self.model_kwarg_keys:
                model_inputs["logits_to_keep"] = logits_to_keep + 1

            logits = model(**model_inputs).logits
            logits = logits[:, :-1, :]
            logits = logits[:, -logits_to_keep:, :]
            logits = logits / self.temperature

            completion_ids = input_ids_batch[:, -logits_to_keep:]
            logps = selective_log_softmax(logits, completion_ids)
            all_logps.append(logps)

            if compute_entropy:
                with torch.no_grad():
                    entropies = entropy_from_logits(logits)
                all_entropies.append(entropies)

        logps = torch.cat(all_logps, dim=0)
        entropies = torch.cat(all_entropies, dim=0) if compute_entropy else None
        return logps, entropies

    def _get_student_tokenizer(self):
        if isinstance(self.processing_class, ProcessorMixin):
            return self.processing_class.tokenizer
        return self.processing_class

    def _is_qwen_pair(self) -> bool:
        teacher_id = (self.teacher_model_id or "").lower()
        student_id = str(getattr(self.model.config, "_name_or_path", "") or "").lower()
        return "qwen" in teacher_id and "qwen" in student_id

    def _ensure_tokenizer_compatibility(self, teacher_tokenizer):
        student_tokenizer = self._get_student_tokenizer()
        if teacher_tokenizer is None or student_tokenizer is None:
            return
        is_qwen = self._is_qwen_pair()
        if getattr(teacher_tokenizer, "vocab_size", None) != getattr(student_tokenizer, "vocab_size", None):
            raise ValueError("Teacher tokenizer vocab_size does not match the student tokenizer.")
        teacher_vocab = teacher_tokenizer.get_vocab()
        student_vocab = student_tokenizer.get_vocab()
        if teacher_vocab != student_vocab:
            raise ValueError("Teacher tokenizer vocab does not match the student tokenizer.")
        def _normalized_special_tokens_map(tokenizer):
            tokens_map = getattr(tokenizer, "special_tokens_map", None)
            if tokens_map is None:
                return None
            if is_qwen:
                return {key: value for key, value in tokens_map.items() if key != "eos_token"}
            return tokens_map
        if _normalized_special_tokens_map(teacher_tokenizer) != _normalized_special_tokens_map(student_tokenizer):
            raise ValueError("Teacher tokenizer special tokens do not match the student tokenizer.")
        if not is_qwen and getattr(teacher_tokenizer, "chat_template", None) != getattr(student_tokenizer, "chat_template", None):
            raise ValueError("Teacher tokenizer chat_template does not match the student tokenizer.")
        test_text = "Tokenization compatibility check."
        teacher_ids = teacher_tokenizer.encode(test_text, add_special_tokens=False)
        student_ids = student_tokenizer.encode(test_text, add_special_tokens=False)
        if teacher_ids != student_ids:
            raise ValueError("Teacher tokenizer encoding differs from the student tokenizer.")

    def _init_teacher(self):
        from unsloth import FastLanguageModel
        from unsloth.models.llama import original_apply_qkv, original_apply_o

        max_seq_length = getattr(self.model, "max_seq_length", None)
        if max_seq_length is None:
            max_seq_length = getattr(self.model.config, "max_position_embeddings", None)
        if max_seq_length is None:
            max_seq_length = 2048

        load_in_4bit = bool(getattr(self.model, "is_loaded_in_4bit", False))
        load_in_8bit = bool(getattr(self.model, "is_loaded_in_8bit", False))
        device_map = {"": self.teacher_device} if self.teacher_device else "sequential"
        prev_compile_disable = os.environ.get("UNSLOTH_COMPILE_DISABLE")
        os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"
        try:
            teacher_model, teacher_tokenizer = FastLanguageModel.from_pretrained(
                model_name=self.teacher_model_id,
                max_seq_length=max_seq_length,
                load_in_4bit=load_in_4bit,
                load_in_8bit=load_in_8bit,
                fast_inference=False,
                device_map=device_map,
            )
        finally:
            if prev_compile_disable is None:
                os.environ.pop("UNSLOTH_COMPILE_DISABLE", None)
            else:
                os.environ["UNSLOTH_COMPILE_DISABLE"] = prev_compile_disable
        self._ensure_tokenizer_compatibility(teacher_tokenizer)

        lora_path = self._resolve_teacher_lora_path(self.teacher_lora_path)
        if lora_path:
            from peft import PeftModel
            teacher_model = PeftModel.from_pretrained(
                teacher_model,
                lora_path,
                is_trainable=False,
            )
        student_tokenizer = self._get_student_tokenizer()
        eos_ids = []
        is_qwen = self._is_qwen_pair()
        if is_qwen:
            for tokenizer in (teacher_tokenizer, student_tokenizer):
                if tokenizer is None:
                    continue
                eos = getattr(tokenizer, "eos_token_id", None)
                if eos is None:
                    continue
                if isinstance(eos, (list, tuple, set)):
                    eos_ids.extend(list(eos))
                else:
                    eos_ids.append(eos)
            eos_ids = sorted(set(eos_ids))
            if eos_ids:
                self._eos_token_ids = eos_ids if len(eos_ids) > 1 else eos_ids[0]
                if hasattr(self, "generation_config"):
                    self.generation_config.eos_token_id = eos_ids if len(eos_ids) > 1 else eos_ids[0]
        teacher_model.eval()
        if hasattr(teacher_model, "for_inference"):
            teacher_model.for_inference()
        for param in teacher_model.parameters():
            param.requires_grad_(False)
        base_model = teacher_model.get_base_model() if hasattr(teacher_model, "get_base_model") else teacher_model
        layers = []
        if hasattr(base_model, "model") and hasattr(base_model.model, "layers"):
            layers = base_model.model.layers
        elif hasattr(base_model, "layers"):
            layers = base_model.layers
        for layer in layers:
            attn = getattr(layer, "self_attn", None)
            if attn is None:
                continue
            if not hasattr(attn, "apply_qkv"):
                attn.apply_qkv = original_apply_qkv
            if not hasattr(attn, "apply_o"):
                attn.apply_o = original_apply_o
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
        input_ids_batch = input_ids.to(teacher_device)
        attention_mask_batch = attention_mask.to(teacher_device)

        model_inputs = {"input_ids": input_ids_batch, "attention_mask": attention_mask_batch}
        if image_grid_thw is not None and pixel_values is not None:
            model_inputs["image_grid_thw"] = image_grid_thw.to(teacher_device)
            model_inputs["pixel_values"] = pixel_values.to(teacher_device)
        elif pixel_values is not None:
            model_inputs["pixel_values"] = pixel_values.to(teacher_device)
        if pixel_attention_mask is not None:
            model_inputs["pixel_attention_mask"] = pixel_attention_mask.to(teacher_device)
        if image_sizes is not None:
            model_inputs["image_sizes"] = image_sizes.to(teacher_device)
        if "logits_to_keep" in self._teacher_model_kwarg_keys:
            model_inputs["logits_to_keep"] = logits_to_keep + 1

        logits = teacher(**model_inputs).logits
        logits = logits[:, :-1, :]
        logits = logits[:, -logits_to_keep:, :]
        logits = logits / self.ivo_beta
        values = self.ivo_beta * torch.logsumexp(logits, dim=-1)
        return values.to(input_ids.device)

    def _generate_and_score_completions(self, inputs):
        self._ensure_vllm_generate_lora_request()
        outputs = super()._generate_and_score_completions(inputs)
        if self._teacher is None or self.teacher_beta <= 0.0:
            return outputs
        from trl.extras.profiling import profiling_context

        prompt_ids, prompt_mask = outputs["prompt_ids"], outputs["prompt_mask"]
        completion_ids, completion_mask = outputs["completion_ids"], outputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)

        with profiling_context(self, "teacher_values"):
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

    @staticmethod
    def _ivo_loss_impl(
        per_token_logps,
        old_per_token_logps,
        completion_mask,
        soft_label,
        num_generations,
    ):
        token_diff = (per_token_logps - old_per_token_logps) * completion_mask
        diff_cum = token_diff.cumsum(dim=1)
        group_diff = diff_cum.view(-1, num_generations, diff_cum.shape[1])
        group_lse = group_diff.float().logsumexp(dim=1).to(diff_cum.dtype)
        per_token_loss = -soft_label.view(-1, num_generations, 1) * (group_diff - group_lse.unsqueeze(1))
        per_token_loss = per_token_loss.view(-1, diff_cum.shape[1])
        token_count = completion_mask.sum().clamp(min=1.0)
        loss = (per_token_loss * completion_mask).sum() / token_count
        return loss

    @staticmethod
    def _ivo_loss_impl_teacher(
        per_token_logps,
        old_per_token_logps,
        completion_mask,
        soft_label,
        num_generations,
        teacher_psi,
        teacher_scale,
    ):
        token_diff = (per_token_logps - old_per_token_logps) * completion_mask
        diff_cum = token_diff.cumsum(dim=1)
        diff_cum = diff_cum - teacher_scale * teacher_psi
        group_diff = diff_cum.view(-1, num_generations, diff_cum.shape[1])
        group_lse = group_diff.float().logsumexp(dim=1).to(diff_cum.dtype)
        per_token_loss = -soft_label.view(-1, num_generations, 1) * (group_diff - group_lse.unsqueeze(1))
        per_token_loss = per_token_loss.view(-1, diff_cum.shape[1])
        token_count = completion_mask.sum().clamp(min=1.0)
        loss = (per_token_loss * completion_mask).sum() / token_count
        return loss

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The IVOTrainer does not support returning outputs")

        with torch.inference_mode(False):
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
            scores = scores.to(per_token_logps.device, copy=True)

            if self.normalized_softlabel:
                soft_label = torch.softmax(
                    scores.view(-1, self.num_generations) / self.ivo_beta,
                    dim=1,
                ).view(-1)
            else:
                soft_label = torch.exp(scores / self.ivo_beta)

            completion_mask = completion_mask.to(per_token_logps.dtype, copy=True)
            soft_label = soft_label.to(per_token_logps.dtype, copy=True)

            teacher_psi = inputs.get("teacher_psi")
            if self.teacher_beta > 0.0:
                if teacher_psi is None:
                    raise ValueError("IVOTrainer requires `teacher_psi` when teacher_beta > 0.")
                teacher_psi = teacher_psi.to(per_token_logps.dtype, copy=True)
                teacher_scale = self.teacher_beta / self.ivo_beta
                loss = self._ivo_loss_impl_teacher(
                    per_token_logps,
                    old_per_token_logps,
                    completion_mask,
                    soft_label,
                    self.num_generations,
                    teacher_psi,
                    teacher_scale,
                )
            else:
                loss = self._ivo_loss_impl(
                    per_token_logps,
                    old_per_token_logps,
                    completion_mask,
                    soft_label,
                    self.num_generations,
                )

            if self.beta != 0.0:
                ref_per_token_logps = inputs.get("ref_per_token_logps")
                if ref_per_token_logps is None:
                    raise ValueError("IVOTrainer requires `ref_per_token_logps` when beta != 0.0.")
                ref_per_token_logps = ref_per_token_logps.to(per_token_logps.dtype, copy=True)
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
