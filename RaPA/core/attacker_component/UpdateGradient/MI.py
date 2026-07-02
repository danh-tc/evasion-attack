import numpy as np
import scipy.stats as st
from .BaseUpdateGradient import BaseUpdateGradient
import torch
from core.registry import GRADIENT_REGISTRY
@GRADIENT_REGISTRY.register_module()
class MI(BaseUpdateGradient):
    def __init__(self, **cfg):
        self.cfg = cfg
        self.mu = cfg.get('mu', 1.0)
        self.last_grad = 0
    def __call__(self, grad):
        g = self.mu * self.last_grad + grad / torch.sum(torch.abs(grad), dim=[1,2,3], keepdim=True)
        self.last_grad = g
        return g
