import argparse
import linecache
import logging
import os
import random
import shutil
import subprocess
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from transformers import (DataCollatorForLanguageModeling, PreTrainedModel,
                          T5Config, T5ForConditionalGeneration, Trainer,
                          TrainingArguments)
from transformers.modeling_outputs import Seq2SeqLMOutput
from transformers.optimization import (AdamW,
                                       get_constant_schedule_with_warmup,
                                       get_linear_schedule_with_warmup)
from transformers.trainer_pt_utils import (DistributedTensorGatherer,
                                           nested_concat)
from transformers.trainer_utils import EvalPrediction, PredictionOutput

from t5chem import SimpleTokenizer, T5ForProperty, data_collator


class MultiTaskTrainer(Trainer):
    """
    Save model weights based on validation error.
    """
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.min_eval_loss: float = float('inf')

    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        eval_dataloader: DataLoader = self.get_eval_dataloader(eval_dataset)
        output: PredictionOutput = self.prediction_loop(
            eval_dataloader,
            description="Evaluation",
            # No point gathering the predictions if there are no metrics, otherwise we defer to
            # self.args.prediction_loss_only
            prediction_loss_only=True if self.compute_metrics is None else None,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )
        self.log(output.metrics) # type: ignore
        cur_loss: float = output.metrics['eval_loss'] # type: ignore
        if self.min_eval_loss >= cur_loss:
            self.min_eval_loss = cur_loss
            for f in Path(self.args.output_dir).glob('best_cp-*'):
                shutil.rmtree(f)
            output_dir: str = os.path.join(self.args.output_dir, f"best_cp-{self.state.global_step}")
            self.save_model(output_dir)
        return output.metrics # type: ignore

    def prediction_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> PredictionOutput:
        """
        Prediction/evaluation loop, shared by :obj:`Trainer.evaluate()` and :obj:`Trainer.predict()`.
        Works both with or without labels.
        """
        prediction_loss_only = (
            prediction_loss_only if prediction_loss_only is not None else self.args.prediction_loss_only
        )

        model = self.model

        batch_size = dataloader.batch_size
        num_examples = self.num_examples(dataloader)
        losses_host = None
        preds_host = None
        labels_host = None

        world_size = 1

        eval_losses_gatherer = DistributedTensorGatherer(world_size, num_examples, make_multiple_of=batch_size)
        if not prediction_loss_only:
            preds_gatherer = DistributedTensorGatherer(world_size, num_examples)
            labels_gatherer = DistributedTensorGatherer(world_size, num_examples)

        model.eval()

        if self.args.past_index >= 0:
            self._past = None

        self.callback_handler.eval_dataloader = dataloader

        for step, inputs in enumerate(dataloader):
            loss, logits, labels = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)
            if loss is not None:
                losses = loss.repeat(batch_size) # type: ignore
                losses_host = losses if losses_host is None else torch.cat((losses_host, losses), dim=0) # type: ignore
            if logits is not None:
                # preds_host = logits if preds_host is None else nested_concat(preds_host, logits, padding_index=-100)
                # logits = torch.stack(logits).unsqueeze(0)
                # logits_reduced = torch.argmax(logits, dim=-1) if (len(logits.size())>1 and logits.size()[-1]>2) else logits
                preds_host = logits if preds_host is None else nested_concat(preds_host, logits, padding_index=-100)
            if labels is not None:
                labels_host = labels if labels_host is None else nested_concat(labels_host, labels, padding_index=-100)
            self.control = self.callback_handler.on_prediction_step(self.args, self.state, self.control)

            # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
            if self.args.eval_accumulation_steps is not None and (step + 1) % self.args.eval_accumulation_steps == 0:
                eval_losses_gatherer.add_arrays(self._gather_and_numpify(losses_host, "eval_losses"))
                if not prediction_loss_only:
                    preds_gatherer.add_arrays(self._gather_and_numpify(preds_host, "eval_preds"))
                    labels_gatherer.add_arrays(self._gather_and_numpify(labels_host, "eval_label_ids"))

                # Set back to None to begin a new accumulation
                losses_host, preds_host, labels_host = None, None, None

        if self.args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        # Gather all remaining tensors and put them back on the CPU
        eval_losses_gatherer.add_arrays(self._gather_and_numpify(losses_host, "eval_losses"))
        if not prediction_loss_only:
            preds_gatherer.add_arrays(self._gather_and_numpify(preds_host, "eval_preds"))
            labels_gatherer.add_arrays(self._gather_and_numpify(labels_host, "eval_label_ids"))

        eval_loss = eval_losses_gatherer.finalize()
        preds = preds_gatherer.finalize() if not prediction_loss_only else None
        label_ids = labels_gatherer.finalize() if not prediction_loss_only else None

        if self.compute_metrics is not None and preds is not None and label_ids is not None:
            metrics = self.compute_metrics(EvalPrediction(predictions=preds, label_ids=label_ids))
        else:
            metrics = {}

        if eval_loss is not None:
            metrics[f"{metric_key_prefix}_loss"] = eval_loss.mean().item()

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return PredictionOutput(predictions=preds, label_ids=label_ids, metrics=metrics)

    # def create_optimizer_and_scheduler(self, num_training_steps: int):
    #     """
    #     Setup the optimizer and the learning rate scheduler.
    #     We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
    #     Trainer's init through :obj:`optimizers`, or subclass and override this method in a subclass.
    #     """
    #     if self.optimizer is None:
    #         no_decay = ["bias", "LayerNorm.weight"]
    #         optimizer_grouped_parameters = [
    #             {
    #                 "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay) and not ('lm_head' in n)],
    #                 "weight_decay": self.args.weight_decay,
    #             },
    #             {
    #                 "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay) and not ('lm_head' in n)],
    #                 "weight_decay": 0.0,
    #             },
    #             {
    #                 "params": [p for n, p in self.model.named_parameters() if 'lm_head' in n],
    #                 'lr': self.args.learning_rate * 0.1,
    #             },
    #         ]
    #         self.optimizer = AdamW(
    #             optimizer_grouped_parameters,
    #             lr=self.args.learning_rate,
    #             betas=(self.args.adam_beta1, self.args.adam_beta2),
    #             eps=self.args.adam_epsilon,
    #         )
    #     if self.lr_scheduler == 'constant':
    #         self.lr_scheduler = get_constant_schedule_with_warmup(
    #             self.optimizer, num_warmup_steps=self.args.warmup_steps
    #         )

    #     elif self.lr_scheduler == 'cosine':
    #         self.lr_scheduler = get_cosine_schedule_with_warmup(
    #             self.optimizer,
    #             num_warmup_steps=self.args.warmup_steps,
    #             num_training_steps=num_training_steps,
    #         )

    #     else:
    #         self.lr_scheduler = get_linear_schedule_with_warmup(
    #             self.optimizer,
    #             num_warmup_steps=self.args.warmup_steps,
    #             num_training_steps=num_training_steps,
    #         )

class MultiTaskDataset(Dataset):
    def __init__(
        self,
        tokenizer,
        data_dir,
        type_path: str="train",
    ) -> None:
        super().__init__()
        
        self.task_types = ["Product", "Reactants", "Reagents", "Classification", "Yield"]
        self._source_path = []
        self._target_path = []
        for task in self.task_types:
            self._source_path.append(os.path.join(data_dir, task, type_path + ".source"))
            self._target_path.append(os.path.join(data_dir, task, type_path + ".target"))
            self._len_source: int = int(subprocess.check_output("wc -l " + self._source_path[-1], shell=True).split()[0])
            self._len_target: int = int(subprocess.check_output("wc -l " + self._target_path[-1], shell=True).split()[0])
            assert self._len_source == self._len_target, "Source file and target file don't match!"
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return self._len_source

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        inputs = {}
        for i,task in enumerate(self.task_types):
            source_line: str = linecache.getline(self._source_path[i], idx + 1).strip()
            source_sample: BatchEncoding = self.tokenizer(
                            task+':'+source_line,
                            max_length=400,
                            padding="do_not_pad",
                            truncation=True,
                            return_tensors='pt',
                        )
            target_line: str = linecache.getline(self._target_path[i], idx + 1).strip()
            if task in ("Classification","Yield"):
                try:
                    target_value: float = float(target_line)
                    target_ids: torch.Tensor = torch.Tensor([target_value])
                except TypeError:
                    print("The target should be a number, \
                            not {}".format(target_line))
                    raise AssertionError
            else:
                target_sample: BatchEncoding = self.tokenizer(
                                target_line,
                                max_length=300,
                                padding="do_not_pad",
                                truncation=True,
                                return_tensors='pt',
                            )
                target_ids = target_sample["input_ids"].squeeze(0)
            source_ids: torch.Tensor = source_sample["input_ids"].squeeze(0)
            src_mask: torch.Tensor = source_sample["attention_mask"].squeeze(0)
            inputs[task] = {"input_ids": source_ids, "attention_mask": src_mask, "decoder_input_ids": target_ids}
        return inputs
                

    def sort_key(self, ex) -> int:
        """ Sort using length of source sentences. """
        return len(ex['Classification']['input_ids'])

def dummy_metrics(model_output: PredictionOutput) -> Dict[str, float]:
    # mask = (model_output.predictions==-100)
    # masked_preds = np.ma.masked_array(model_output.predictions, mask=mask)
    prod_acc, rct_acc, rgt_acc, cls_acc, mse_yield = model_output.predictions.mean(0)
    return {
        'product_acc': prod_acc.item(),
        'reactants_acc': rct_acc.item(),
        'reagents_acc': rgt_acc.item(),
        'classification_acc': cls_acc.item(),
        'mse_loss': mse_yield.item(),
    }

def MT_collator(batches, pad_token_id: int):
    ex = batches[0]
    # Product, Reactants, Reagents, Classification, Yield
    whole_batch = {}
    for task in ex.keys():
        task_batch = {}
        for key in ex[task].keys():
            if 'mask' in key:
                padding_value = 0
            else:
                padding_value = pad_token_id
            task_batch[key] = pad_sequence([x[task][key] for x in batches],
                                            batch_first=True,
                                            padding_value=padding_value)
        source_ids, source_mask, y = \
            task_batch["input_ids"], task_batch["attention_mask"], task_batch["decoder_input_ids"]
        whole_batch[task] = {'input_ids': source_ids, 'attention_mask': source_mask,
            'labels': y}
    return {"input_dict":whole_batch, 'labels':y}

class T5ForMultiTask(nn.Module):
    def __init__(self, pretrain_path, seq2seqW=1):
        super().__init__()
        self.Seq2seqModel = T5ForConditionalGeneration.from_pretrained(pretrain_path)
        self.ClassificationModel = T5ForProperty.from_pretrained(pretrain_path, head_type='classification')
        self.RegressionModel = T5ForProperty.from_pretrained(pretrain_path, head_type='regression')
        self.seq2seqW = seq2seqW

        # tie weights
        self.shared = self.Seq2seqModel.shared
        self.encoder = self.Seq2seqModel.encoder
        self.decoder = self.Seq2seqModel.decoder

        self.RegressionModel.shared = self.ClassificationModel.shared = self.shared
        self.RegressionModel.encoder = self.ClassificationModel.encoder = self.encoder
        self.RegressionModel.decoder = self.ClassificationModel.decoder = self.decoder

        # Assign correct task type
        self.Seq2seqModel.config.task_type = 'mixed'
        self.ClassificationModel.config.task_type = 'classification'
        self.RegressionModel.config.task_type = 'regression'

    def forward(self,input_dict,labels):
    #         pdb.set_trace()
        loss = 0
        # product
        outputs_prod = self.Seq2seqModel(**input_dict['Product'])
        preds = torch.argmax(outputs_prod['logits'], dim=-1)
        label = input_dict['Product']['labels']
        prod_acc = torch.all(preds==label,1) #.sum()
        loss += self.seq2seqW*outputs_prod["loss"] if isinstance(outputs_prod, dict) else outputs_prod[0]

        # reactants
        outputs_rct = self.Seq2seqModel(**input_dict['Reactants'])
        preds = torch.argmax(outputs_rct['logits'], dim=-1)
        label = input_dict['Reactants']['labels']
        rct_acc = torch.all(preds==label,1) #.sum()
        loss += self.seq2seqW*outputs_rct["loss"] if isinstance(outputs_rct, dict) else outputs_rct[0]
        
        # reagents
        outputs_rgt = self.Seq2seqModel(**input_dict['Reagents'])
        preds = torch.argmax(outputs_rgt['logits'], dim=-1)
        label = input_dict['Reagents']['labels']
        rgt_acc = torch.all(preds==label,1) #.sum()
        loss += self.seq2seqW*outputs_rgt["loss"] if isinstance(outputs_rgt, dict) else outputs_rgt[0]
        
        # classification
        outputs_cls = self.ClassificationModel(**input_dict['Classification'])
        preds = outputs_cls['logits']
        label = input_dict['Classification']['labels'].to(outputs_cls['logits']).squeeze()
        cls_acc = preds==label #(preds==label).sum()
        loss += outputs_cls["loss"] if isinstance(outputs_cls, dict) else outputs_cls[0]

        # regression
        outputs_yd = self.RegressionModel(**input_dict['Yield'])
        preds = outputs_yd["logits"]
        label = input_dict['Yield']['labels'].squeeze()
        mse_sum = (preds-label)**2 #.sum()
        loss += outputs_yd["loss"] if isinstance(outputs_yd, dict) else outputs_yd[0]
        return Seq2SeqLMOutput(
            loss=loss,
            logits=torch.stack([prod_acc, rct_acc, rgt_acc, cls_acc, mse_sum],1),
        )

if __name__ == "__main__":
    model = T5ForMultiTask("models/pretrain/simple/", seq2seqW=1.5)
    tokenizer = SimpleTokenizer(vocab_file="models/pretrain/simple/vocab.pt")
    os.makedirs("models/MultiTaskTrainw1.5/", exist_ok=True)
    tokenizer.save_vocabulary(os.path.join("models/MultiTaskTrainw1.5/", 'vocab.pt'))
    dataset = MultiTaskDataset(
        tokenizer, 
        data_dir="../t5chem_data/USPTO_MT/",
    )
    data_collator_padded = partial(
        MT_collator, pad_token_id=tokenizer.pad_token_id)
    eval_strategy = "steps"
    eval_iter = MultiTaskDataset(
        tokenizer, 
        data_dir="../t5chem_data/USPTO_MT/",
        type_path="val",
    )

    #     if task.output_layer == 'regression':
    #         compute_metrics = CalMSELoss
    #     elif args.task_type == 'pretrain':
    compute_metrics = dummy_metrics  
    #         # We don't want any extra metrics for faster pretraining
    #     else:
    #         compute_metrics = AccuracyMetrics

    training_args = TrainingArguments(
        output_dir="models/MultiTaskTrainw1.5/",
        overwrite_output_dir=True,
        do_train=True,
        evaluation_strategy=eval_strategy,
        num_train_epochs=100,
        per_device_train_batch_size=32,
        logging_steps=5000,
        per_device_eval_batch_size=32,
        save_steps=50000,
        save_total_limit=5,
        learning_rate=5e-4,
        prediction_loss_only=(compute_metrics is None),
    )

    trainer = MultiTaskTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator_padded,
        train_dataset=dataset,
        eval_dataset=eval_iter,
        compute_metrics=compute_metrics,
    )
    # pdb.set_trace()
    trainer.train()
    # print(args)
    # print("logging dir: {}".format(training_args.logging_dir))
    trainer.save_model("models/MultiTaskTrainw1.5/")
