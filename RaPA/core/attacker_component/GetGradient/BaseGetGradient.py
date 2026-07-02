import torch
from core.registry import *

@GRADIENT_REGISTRY.register_module()
class BaseGetGradient:
    def __init__(self, loss_fn, input_transform, **cfg):

        self.loss_fn = loss_fn
        self.input_transform = input_transform
        self.cfg = cfg

    def __call__(self, model, x, delta, label, **kwargs):
        x_adv = x + delta
        logits = model(self.input_transform(x_adv))
        loss = self.loss_fn(logits, label)
        # print(loss)
        grad = torch.autograd.grad(loss, x_adv)[0]
        return grad
        