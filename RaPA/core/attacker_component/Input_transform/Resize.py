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
class Resize(BaseTransform):
    def __init__(self, **cfg):
        super(Resize, self).__init__(**cfg)
        self.resize_size = cfg.get('resize_size', (224, 224))
        self.input_transform = TRANSFORM_REGISTRY.build(self.cfg.get("input_transform", {"type":"BaseTransform"}) )

    def __call__(self, X_in):
        transform = transforms.transforms.Resize(self.resize_size, interpolation=InterpolationMode.BILINEAR)
        x_resize = transform(X_in)
        return self.input_transform(x_resize)