import numpy as np
import scipy.stats as st
from .BaseUpdateGradient import BaseUpdateGradient
import torch
from core.registry import GRADIENT_REGISTRY
import torch.nn.functional as F

@GRADIENT_REGISTRY.register_module()
class TI(BaseUpdateGradient):
    def __init__(self, **cfg):
        self.cfg = cfg
        self.kernlen = cfg.get('kernel_size', 5)
        self.sigma = cfg.get('sigma', 3)

        kernel = self.gkern().astype(np.float32)

        gaussian_kernel = np.stack([kernel, kernel, kernel])
        gaussian_kernel = np.expand_dims(gaussian_kernel, 1)
        self.gaussian_kernel = torch.from_numpy(gaussian_kernel).cuda()
        
        update_gradient_cfg = cfg.get('update_gradient', {"type": "BaseUpdateGradient"})
        self.update_gradient = GRADIENT_REGISTRY.build(update_gradient_cfg)

    def gkern(self):
        kernlen, nsig = self.kernlen, self.sigma
        x = np.linspace(-nsig, nsig, kernlen)
        kern1d = st.norm.pdf(x)
        kernel_raw = np.outer(kern1d, kern1d)
        kernel = kernel_raw / kernel_raw.sum()
        return kernel
    def __call__(self, grad):
        grad = F.conv2d(grad, self.gaussian_kernel, bias=None, stride=1, padding=((self.kernlen-1)//2,(self.kernlen-1)//2), groups=3)
        grad = self.update_gradient(grad)
        return grad
