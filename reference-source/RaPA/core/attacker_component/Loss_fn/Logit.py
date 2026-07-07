
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from PIL import Image
from core.registry import LOSS_FN_REGISTRY

@LOSS_FN_REGISTRY.register_module()
class Logit(nn.Module):
    def __init__(self, **cfg):
        super(Logit, self).__init__()
        self.cfg = cfg
    def forward(self, logits, labels):
        if self.cfg.get("is_targeted", False):
            labels = labels[1].cuda()
            real = logits.gather(1, labels.unsqueeze(1)).squeeze(1)
            logit_dists = (1 * real)
            loss = logit_dists.sum()
        else:
            labels = labels.cuda()
            real = logits.gather(1, labels.unsqueeze(1)).squeeze(1)
            logit_dists = (1 * real)
            loss = logit_dists.sum()
            loss = -loss
        return loss 