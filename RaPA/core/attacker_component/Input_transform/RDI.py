from .BaseTransform import BaseTransform
from core.registry import TRANSFORM_REGISTRY
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms
import torchvision
from torchvision.transforms import InterpolationMode
@TRANSFORM_REGISTRY.register_module()
class RDI(BaseTransform):
    def __init__(self, **cfg):
        super(RDI, self).__init__(**cfg)
        self.resize_ratio = cfg.get("resize_ratio", 340./299.)

    def __call__(self, X_in):
        x_di = X_in 
        img_width=X_in.size()[-1] # B X C X H X W
        enlarged_img_width=int(img_width * self.resize_ratio)
        di_pad_amount=enlarged_img_width-img_width
        di_pad_value=0
        ori_size = x_di.shape[-1]
        rnd = int(torch.rand(1) * di_pad_amount) + ori_size
        x_di = transforms.Resize((rnd, rnd), interpolation=InterpolationMode.NEAREST)(x_di)
        pad_max = ori_size + di_pad_amount - rnd
        pad_left = int(torch.rand(1) * pad_max)
        pad_right = pad_max - pad_left
        pad_top = int(torch.rand(1) * pad_max)
        pad_bottom = pad_max - pad_top
        x_di = F.pad(x_di, (pad_left, pad_right, pad_top, pad_bottom), 'constant', di_pad_value)
        if img_width>64: # For the CIFAR-10 dataset, we skip the image size reduction.
            x_di = transforms.Resize((ori_size, ori_size), interpolation=InterpolationMode.NEAREST)(x_di)
        return x_di