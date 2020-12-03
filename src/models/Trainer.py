import os
import shutil
from pathlib import Path
from transformers import Trainer

class EarlyStopTrainer(Trainer):
    """
    Save model weights based on validation error.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.min_eval_loss = float('inf')

    def evaluate(self, eval_dataset=None):
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        output = self.prediction_loop(eval_dataloader, description="Evaluation")
        self.log(output.metrics)
        if 'eval_mse_loss' in output.metrics:
            cur_loss = output.metrics['eval_mse_loss']
        else:
            cur_loss = output.metrics['eval_loss']
        if self.min_eval_loss >= cur_loss:
            self.min_eval_loss = cur_loss
            for f in Path(self.args.output_dir).glob('best_cp-*'):
                shutil.rmtree(f)
            output_dir = os.path.join(self.args.output_dir, f"best_cp-{self.global_step}")
            self.save_model(output_dir)
        return output.metrics
