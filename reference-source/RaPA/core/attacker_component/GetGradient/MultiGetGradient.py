import torch
from core.registry import *
from .BaseGetGradient import BaseGetGradient

@GRADIENT_REGISTRY.register_module()
class MultiGetGradient(BaseGetGradient):
    def __init__(self, loss_fn, input_transform, **cfg):
        super(MultiGetGradient, self).__init__(loss_fn, input_transform, **cfg)
        self.times = cfg.get('times', 2)
        
    def __call__(self, model, x, delta, label, **kwargs):
        grad = 0
        for _ in range(self.times):
            x_adv = x + delta
            logits = model(self.input_transform(x_adv))
            loss = self.loss_fn(logits, label)
            grad = grad + torch.autograd.grad(loss, delta)[0]
        return grad
        