
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from PIL import Image
from core.registry import LOSS_FN_REGISTRY


@LOSS_FN_REGISTRY.register_module()
class CrossEntropy(nn.Module):
    def __init__(self, **cfg):
        super(CrossEntropy, self).__init__()
        self.cfg = cfg
        self.ce = nn.CrossEntropyLoss()
    def forward(self, logits, labels):
        loss = None
        if self.cfg.get("is_targeted", False):
            labels = labels[1].cuda()
            loss = -self.ce(logits, labels)
        else:
            labels = labels.cuda()
            loss = self.ce(logits, labels)
        return loss