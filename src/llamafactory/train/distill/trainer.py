# Copyright 2025 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/trainer_seq2seq.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
from types import MethodType
from typing import TYPE_CHECKING, Any, Optional, Union
from transformers import AutoModelForCausalLM

import numpy as np
import torch
from transformers import Seq2SeqTrainer
from typing_extensions import override
import deepspeed


#from ...extras import logging
import logging
from ...extras.constants import IGNORE_INDEX
from ..callbacks import SaveProcessorCallback
from ..fp8_utils import configure_fp8_environment, patch_accelerator_for_fp8, verify_fp8_status
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler
#from trl.trainer.utils import prepare_deepspeed


if TYPE_CHECKING:
    from torch.utils.data import Dataset
    from transformers import ProcessorMixin
    from transformers.trainer import PredictionOutput

    from ...hparams import FinetuningArguments, ModelArguments, TrainingArguments


logger = logging.getLogger(__name__)
logger.handlers.clear()
logger.setLevel(logging.DEBUG)

#file_handler = logging.FileHandler(
#    filename="/data/sty/LLaMA-Factory/log.txt",  
#    mode="a",                   
#    encoding="utf-8"             
#)
#logger.addHandler(file_handler)

def padding_sequence(batch_samples):
    max_len = max([s["input_ids"].shape[1] for s in batch_samples])
    pad_token_id = 151643
    
    padded_input_ids, padded_labels, padded_attn = [], [], []
    for sample in batch_samples:
        pad_len = max_len - sample["input_ids"].shape[1]
        
        padded_input = torch.cat([
            sample["input_ids"].squeeze(0), 
            torch.full((pad_len,), pad_token_id, dtype=torch.long,device=sample["input_ids"].device)
        ])
        padded_label = torch.cat([
            sample["labels"].squeeze(0), 
            torch.full((pad_len,), -100, dtype=torch.long,device=sample["labels"].device)
        ])
        padded_attn_mask = torch.cat([
            sample["attention_mask"].squeeze(0), 
            torch.zeros(pad_len, dtype=torch.long,device=sample["attention_mask"].device)
        ])
        
        padded_input_ids.append(padded_input)
        padded_labels.append(padded_label)
        padded_attn.append(padded_attn_mask)
    
    final_batch = {}
    final_batch["input_ids"] = torch.stack(padded_input_ids)
    final_batch["labels"] = torch.stack(padded_labels)
    final_batch["attention_mask"] = torch.stack(padded_attn)
    
    return final_batch

class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    r"""Inherits Seq2SeqTrainer to compute generative metrics such as BLEU and ROUGE."""

    def __init__(
        self,
        teacher_name,
        select_type,
        train_type,
        finetuning_args: "FinetuningArguments",
        processor: Optional["ProcessorMixin"],
        model_args: Optional["ModelArguments"] = None,
        gen_kwargs: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        kwargs["processing_class"] = kwargs.pop("tokenizer")
        # Configure FP8 environment if enabled
        training_args: TrainingArguments = kwargs.get("args")
        if training_args.fp8:
            configure_fp8_environment(training_args)
            if getattr(training_args, "fp8_backend", "auto") == "te":
                patch_accelerator_for_fp8()

        super().__init__(**kwargs)
        if processor is not None:
            # avoid wrong loss under gradient accumulation
            # https://github.com/huggingface/transformers/pull/36044#issuecomment-2746657112
            self.model_accepts_loss_kwargs = False

        self.finetuning_args = finetuning_args
        if gen_kwargs is not None:
            # https://github.com/huggingface/transformers/blob/v4.45.0/src/transformers/trainer_seq2seq.py#L287
            self._gen_kwargs = gen_kwargs

        if processor is not None:
            self.add_callback(SaveProcessorCallback(processor))

        if finetuning_args.use_badam:
            from badam import BAdamCallback, clip_grad_norm_old_version  # type: ignore

            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_old_version, self.accelerator)
            self.add_callback(BAdamCallback)

        if finetuning_args.use_dft_loss:
            from ..trainer_utils import dft_loss_func

            self.compute_loss_func = dft_loss_func

        elif finetuning_args.use_eaft_loss:
            from ..trainer_utils import eaft_loss_func

            self.compute_loss_func = lambda outputs, labels, num_items_in_batch=None: eaft_loss_func(
                outputs, labels, num_items_in_batch, finetuning_args.eaft_alpha
            )


        if training_args.fp8 and hasattr(self, "accelerator"):  # verify FP8 status after trainer initialization
            verify_fp8_status(self.accelerator, training_args)

        self.teacher_model = AutoModelForCausalLM.from_pretrained(teacher_name,torch_dtype=torch.bfloat16).to(f"cuda:{os.environ.get('RANK')}")
        self.teacher_model.eval()
        self.select_type = select_type
        self.train_type = train_type
        #self.teacher_model = prepare_deepspeed(self.teacher_model,1)
        #self.teacher_model = deepspeed.init_inference(model=self.teacher_model)

    def select_sample(self,inputs,model):
        input_ids = inputs["input_ids"]
        labels = inputs["labels"]
        attention_mask = inputs["attention_mask"]
        
        #sep_token_id = self.tokenizer.encode("######", add_special_tokens=False)[0]
        sep_token_id = 77129
        #logger.debug(input_ids.tolist())
        
        batch_samples = []
        for idx in range(input_ids.shape[0]):
            split_samples = []
            
            single_input = input_ids[idx]
            single_label = labels[idx]
            single_attn = attention_mask[idx]

            
            #stytext = self.tokenizer.decode(single_input, skip_special_tokens=True)
            #logger.debug(os.environ.get("RANK")+"--------"+repr(stytext))

            sep_pos = (single_input == sep_token_id).nonzero().squeeze(-1).tolist()
            if not isinstance(sep_pos, list):
                sep_pos = [sep_pos] if sep_pos != -1 else []
            
            if not sep_pos:
                logger.error("???????")
                logger.error(single_input)

            head_end = sep_pos[0]
            head_input = single_input[:head_end]
            head_label = single_label[:head_end]
            head_attn = single_attn[:head_end]

            for i, pos in enumerate(sep_pos):
                sep_end = pos + 1
                if i == len(sep_pos) - 1:
                    tail_input = single_input[sep_end:]
                    tail_label = single_label[sep_end:]
                    tail_attn = single_attn[sep_end:]
                else:
                    tail_input = single_input[sep_end:sep_pos[i+1]]
                    tail_label = single_label[sep_end:sep_pos[i+1]]
                    tail_attn = single_attn[sep_end:sep_pos[i+1]]

                combined_input = torch.cat([head_input, tail_input])
                combined_label = torch.cat([head_label, tail_label])
                combined_attn = torch.cat([head_attn, tail_attn])

                #combined_text = self.tokenizer.decode(combined_input, skip_special_tokens=True)
                #logger.debug(os.environ.get("RANK")+"--------"+str(i)+"---------"+repr(combined_text))
                split_samples.append({
                    "input_ids": combined_input.unsqueeze(0),
                    "labels": combined_label.unsqueeze(0),
                    "attention_mask": combined_attn.unsqueeze(0)
                })
            with torch.no_grad():
                loss_list=[]
                if "ce" in self.select_type:
                    for sample in split_samples:
                        outputs = model(**sample)
                        loss = outputs.loss.item()
                        loss_list.append(loss)
                else:
                    sample = padding_sequence(split_samples)
                    #logger.debug(os.environ.get("RANK")+"-------padding"+str(sample["input_ids"].shape))
                    outputs = model(**sample)
                    teacher_outputs = self.teacher_model(**sample)
                    teacher_logits = teacher_outputs.logits
                    teacher_probs = torch.nn.functional.softmax(teacher_logits, dim=-1)
                    student_log_probs = torch.nn.functional.log_softmax(outputs.logits, dim=-1)
                    kl_div =  torch.nn.functional.kl_div( 
                        student_log_probs,
                        teacher_probs, 
                        reduction="none"  
                    )
                    mask = (sample["labels"] != -100).unsqueeze(-1).to(kl_div.device)
                    kl_div_masked = kl_div * mask
                    loss = kl_div_masked.sum(-1).mean(-1)
                    #logger.debug(os.environ.get("RANK")+"-------"+str(loss.shape))
                    loss_list = loss.tolist()
                    #logger.debug(loss_list)
                    #logger.debug(os.environ.get("RANK")+"-------"+str(len(loss_list)))
                 
                if "max" in self.select_type:
                    selected_val = max(loss_list)
                else:
                    selected_val = max(loss_list)
                selected_idx = loss_list.index(selected_val)  
                selected_sample = split_samples[selected_idx]
                
                batch_samples.append(selected_sample)

        torch.cuda.empty_cache()
        final_batch = padding_sequence(batch_samples)
       
        return final_batch

    @override
    def create_optimizer(self) -> "torch.optim.Optimizer":
        if self.optimizer is None:
            self.optimizer = create_custom_optimizer(self.model, self.args, self.finetuning_args)
        return super().create_optimizer()

    @override
    def create_scheduler(
        self, num_training_steps: int, optimizer: Optional["torch.optim.Optimizer"] = None
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        create_custom_scheduler(self.args, num_training_steps, optimizer)
        return super().create_scheduler(num_training_steps, optimizer)

    @override
    def _get_train_sampler(self, *args, **kwargs) -> Optional["torch.utils.data.Sampler"]:
        if self.finetuning_args.disable_shuffling:
            return torch.utils.data.SequentialSampler(self.train_dataset)

        return super()._get_train_sampler(*args, **kwargs)

    @override
    def compute_loss(self, model, inputs, *args, **kwargs):
        
        selected_inputs = self.select_sample(inputs,model)
        #logger.debug(f"GPU {os.environ.get('RANK')}: {torch.cuda.memory_allocated(int(os.environ.get('RANK')))/1024**2:.2f} MB")
        #logger.debug(os.environ.get("RANK")+"-------"+str(selected_inputs["input_ids"].shape)+"-------"+str(selected_inputs["labels"].shape)+"-------"+str(selected_inputs["attention_mask"].shape))
        
        #logger.debug(os.environ.get("RANK")+"-------mask"+str(selected_inputs["labels"].shape))
        
        outputs = model(**selected_inputs)
        
        if self.train_type == "distill":            
            with torch.no_grad():
                #logger.info(self.teacher_model.device)
                teacher_outputs = self.teacher_model(**selected_inputs)
                teacher_logits = teacher_outputs.logits
                teacher_probs = torch.nn.functional.softmax(teacher_logits, dim=-1)
                
            student_log_probs = torch.nn.functional.log_softmax(outputs.logits, dim=-1)
            
            kl_div =  torch.nn.functional.kl_div( 
                student_log_probs,
                teacher_probs, 
                reduction="none"  
            )
            
            mask = (selected_inputs["labels"] != -100).unsqueeze(-1).to(kl_div.device)
            #logger.debug(os.environ.get("RANK")+"-------mask"+str(mask.shape))
            kl_div_masked = kl_div * mask
            
            loss = kl_div_masked.sum(-1).mean() 
            #logger.debug(f"GPU {os.environ.get('RANK')}: {torch.cuda.memory_allocated(int(os.environ.get('RANK')))/1024**2:.2f} MB")
            #loss = self.label_smoother(outputs, labels, shift_labels=True)
        else:
            loss =  outputs.loss
        
        if (
            self.args.average_tokens_across_devices
            and (self.model_accepts_loss_kwargs or self.compute_loss_func)
        ):
            loss *= self.accelerator.num_processes if self.args.n_gpu <= 1 else self.args.n_gpu
            #logger.debug(self.accelerator.num_processes)
        
        torch.cuda.empty_cache()
        return loss
        
    @override
    def prediction_step(
        self,
        model: "torch.nn.Module",
        inputs: dict[str, Union["torch.Tensor", Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[list[str]] = None,
        **gen_kwargs,
    ) -> tuple[Optional[float], Optional["torch.Tensor"], Optional["torch.Tensor"]]:
        r"""Remove the prompt part in the generated tokens.

        Subclass and override to inject custom behavior.
        """
        if self.args.predict_with_generate:  # do not pass labels to model when generate
            labels = inputs.pop("labels", None)
        else:
            labels = inputs.get("labels")

        loss, generated_tokens, _ = super().prediction_step(
            model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys, **gen_kwargs
        )
        if generated_tokens is not None and self.args.predict_with_generate:
            generated_tokens[:, : inputs["input_ids"].size(-1)] = self.processing_class.pad_token_id
            generated_tokens = generated_tokens.contiguous()

        return loss, generated_tokens, labels

    def save_predictions(
        self, dataset: "Dataset", predict_results: "PredictionOutput", skip_special_tokens: bool = True
    ) -> None:
        r"""Save model predictions to `output_dir`.

        A custom behavior that not contained in Seq2SeqTrainer.
        """
        if not self.is_world_process_zero():
            return

        output_prediction_file = os.path.join(self.args.output_dir, "generated_predictions.jsonl")
        logger.info_rank0(f"Saving prediction results to {output_prediction_file}")

        labels = np.where(
            predict_results.label_ids != IGNORE_INDEX, predict_results.label_ids, self.processing_class.pad_token_id
        )
        preds = np.where(
            predict_results.predictions != IGNORE_INDEX,
            predict_results.predictions,
            self.processing_class.pad_token_id,
        )

        for i in range(len(preds)):
            pad_len = np.nonzero(preds[i] != self.processing_class.pad_token_id)[0]
            if len(pad_len):  # move pad token to last
                preds[i] = np.concatenate((preds[i][pad_len[0] :], preds[i][: pad_len[0]]), axis=-1)

        decoded_inputs = self.processing_class.batch_decode(dataset["input_ids"], skip_special_tokens=False)
        decoded_preds = self.processing_class.batch_decode(preds, skip_special_tokens=skip_special_tokens)
        decoded_labels = self.processing_class.batch_decode(labels, skip_special_tokens=skip_special_tokens)

        with open(output_prediction_file, "w", encoding="utf-8") as f:
            for text, pred, label in zip(decoded_inputs, decoded_preds, decoded_labels):
                f.write(json.dumps({"prompt": text, "predict": pred, "label": label}, ensure_ascii=False) + "\n")

    