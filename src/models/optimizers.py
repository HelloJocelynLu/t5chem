from copy import deepcopy
from transformers.optimization import AdamW

class ModelUpdateAdamW(AdamW):
    def __init__(self, params, model, lr=1e-3, betas=(0.9, 0.999), eps=1e-6, weight_decay=0):
        super().__init__(params, lr, betas, eps, weight_decay)
        self.model = model

    def step(self, closure=None):
        loss = super().step(closure)
        if hasattr(self.model, "update"):
            self.model.update()
        return loss
