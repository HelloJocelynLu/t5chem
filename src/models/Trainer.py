import os
import shutil
import torch
from pathlib import Path
from transformers import Trainer
from transformers.optimization import get_linear_schedule_with_warmup, \
        get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup
from transformers.trainer_pt_utils import (
    DistributedTensorGatherer,
    nested_concat,
)
from transformers.trainer_utils import (
    EvalPrediction,
    PredictionOutput,
)
from .optimizers import ModelUpdateAdamW

class EarlyStopTrainer(Trainer):
    """
    Save model weights based on validation error.
    """
    def __init__(self, train_eval_dataset=None, **kwargs):
        super().__init__(**kwargs)
        self.train_eval_dataset = train_eval_dataset
        self.min_eval_loss = float('inf')

    def evaluate(self, eval_dataset=None, ignore_keys=None):
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        if self.train_eval_dataset:
            train_eval_dataloader = self.get_eval_dataloader(self.train_eval_dataset)
            output_ = self.prediction_loop(
                train_eval_dataloader,
                description="Evaluation",
                # No point gathering the predictions if there are no metrics, otherwise we defer to
                # self.args.prediction_loss_only
                prediction_loss_only=True if self.compute_metrics is None else None,
                ignore_keys=ignore_keys,
                metric_key_prefix="train_eval",
            )
            self.log(output_.metrics)
        output = self.prediction_loop(
            eval_dataloader,
            description="Evaluation",
            # No point gathering the predictions if there are no metrics, otherwise we defer to
            # self.args.prediction_loss_only
            prediction_loss_only=True if self.compute_metrics is None else None,
            ignore_keys=ignore_keys
        )
        self.log(output.metrics)
        if 'eval_mse_loss' in output.metrics:
            cur_loss = output.metrics['eval_mse_loss']
#        elif 'eval_accuracy' in output.metrics:
#            cur_loss = -output.metrics['eval_accuracy']
        else:
            cur_loss = output.metrics['eval_loss']
        if self.min_eval_loss >= cur_loss:
            self.min_eval_loss = cur_loss
            for f in Path(self.args.output_dir).glob('best_cp-*'):
                shutil.rmtree(f)
            output_dir = os.path.join(self.args.output_dir, f"best_cp-{self.state.global_step}")
            self.save_model(output_dir)
        return output.metrics

    def prediction_loop(
        self,
        dataloader,
        description,
        prediction_loss_only = None,
        ignore_keys = None,
        metric_key_prefix = "eval",
    ):
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
                losses = loss.repeat(batch_size)
                losses_host = losses if losses_host is None else torch.cat((losses_host, losses), dim=0)
            if logits is not None:
                # preds_host = logits if preds_host is None else nested_concat(preds_host, logits, padding_index=-100)
                logits = logits[0]
                logits_reduced = torch.argmax(logits, axis=-1) if len(logits.size())>1 else logits
                preds_host = logits_reduced if preds_host is None else nested_concat(preds_host, logits_reduced, padding_index=-100)
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

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        """
        Setup the optimizer and the learning rate scheduler.
        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through :obj:`optimizers`, or subclass and override this method in a subclass.
        """
        if self.optimizer is None:
            no_decay = ["bias", "LayerNorm.weight"]
            if hasattr(self.model, "shadow"):
                model = self.model.model
            else:
                model = self.model
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                },
            ]
            self.optimizer = ModelUpdateAdamW(
                optimizer_grouped_parameters,
                model = self.model,
                lr=self.args.learning_rate,
                betas=(self.args.adam_beta1, self.args.adam_beta2),
                eps=self.args.adam_epsilon,
            )
        if self.lr_scheduler == 'constant':
            self.lr_scheduler = get_constant_schedule_with_warmup(
                self.optimizer, num_warmup_steps=self.args.warmup_steps
            )

        elif self.lr_scheduler == 'cosine':
            self.lr_scheduler = get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=self.args.warmup_steps,
                num_training_steps=num_training_steps,
            )

        else:
            self.lr_scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=self.args.warmup_steps,
                num_training_steps=num_training_steps,
            )

    def _save(self, output_dir = None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        if hasattr(self.model, "shadow"):
            model = self.model.shadow
        else:
            model = self.model
        model.save_pretrained(output_dir)
        if self.tokenizer is not None and self.is_world_process_zero():
            self.tokenizer.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))
