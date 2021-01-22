from copy import deepcopy
from transformers.optimization import AdamW

class EMAAdamW(AdamW):
    def __init__(self, params, model, lr=1e-3, betas=(0.9, 0.999), eps=1e-6, weight_decay=0, ema=0.999):
        super().__init__(params, lr, betas, eps, weight_decay)
        self.ema = ema
        self.shadow_model = deepcopy(model)
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.shadow_model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
            {
                "params": [p for n, p in self.shadow_model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        tmp_optim = AdamW(optimizer_grouped_parameters)
        self.shadow_param_groups = tmp_optim.param_groups
        self.deactivated = (ema < 0)

    def step(self, closure=None):
        loss = super().step(closure)

        if not self.deactivated:
            for shadow, m_params in zip(self.shadow_param_groups, self.param_groups):
                for p_shadow, p_model in zip(shadow['params'], m_params['params']):
                    p_shadow.data.add_((1 - self.ema) * (p_model.data - p_shadow.data))

        return loss
