import os
import shutil
import torch
from pathlib import Path
from transformers import Trainer
from transformers.optimization import get_linear_schedule_with_warmup, \
        get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup
from .optimizers import ModelUpdateAdamW

class EarlyStopTrainer(Trainer):
    """
    Save model weights based on validation error.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.min_eval_loss = float('inf')

    def evaluate(self, eval_dataset=None, ignore_keys=None):
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        output = self.prediction_loop(
            eval_dataloader,
            description="Evaluation",
            # No point gathering the predictions if there are no metrics, otherwise we defer to
            # self.args.prediction_loss_only
            prediction_loss_only=True if self.compute_metrics is None else None,
            ignore_keys=ignore_keys,
        )
        self.log(output.metrics)
        if 'eval_mse_loss' in output.metrics:
            cur_loss = output.metrics['eval_mse_loss']
        elif 'eval_accuracy' in output.metrics:
            cur_loss = -output.metrics['eval_accuracy']
        else:
            cur_loss = output.metrics['eval_loss']
        if self.min_eval_loss >= cur_loss:
            self.min_eval_loss = cur_loss
            for f in Path(self.args.output_dir).glob('best_cp-*'):
                shutil.rmtree(f)
            output_dir = os.path.join(self.args.output_dir, f"best_cp-{self.state.global_step}")
            self.save_model(output_dir)
        return output.metrics


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
