from core.registry import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Callable

@ATTACK_REGISTRY.register_module()
class BaseAttacker:
    def __init__(self, **cfg):
        self.src_model = cfg.get("src_model")
        self.is_targeted = cfg.get("is_targeted")
        self.max_iterations = cfg.get("max_iterations", 300)
        self.epsilon = cfg.get("epsilon", 16/255)
        self.alpha = cfg.get("alpha", 2/255)
        self.use_tqdm = cfg.get("use_tqdm", False)
        tester_cfg = cfg.get("tester_cfg")
        assert tester_cfg is not None, "tester_cfg is not specified"
        tester_cfg.update({"max_iter": self.max_iterations})
        
        self.tester = TESTER_REGISTRY.build(tester_cfg)
        if tester_cfg.get("test_interval") is None:
            self.tester.test_interval = self.max_iterations
        self.work_dir = cfg.get("work_dir", None)
        self.cfg = cfg
    def __call__(self, data, label, **kwargs):
        raise NotImplementedError
        # return self.attack(data, label, **kwargs)
    
    def build_before_attack(self):
        self.input_transform = TRANSFORM_REGISTRY.build(self.cfg.get("input_transform"))
        self.loss_fn = LOSS_FN_REGISTRY.build(self.cfg.get("loss_fn"))

        get_gradient_cfg = self.cfg.get("get_gradient")
        get_gradient_cfg.update({"loss_fn": self.loss_fn, "input_transform": self.input_transform})
        self.get_gradient = GRADIENT_REGISTRY.build(get_gradient_cfg)

        if self.cfg.get("update_gradient") is None:
            self.cfg["update_gradient"] = {"type": "BaseUpdateGradient"}
            print("Warning: update_gradient is not specified, using BaseUpdateGradient")
        self.update_gradient = GRADIENT_REGISTRY.build(self.cfg.get("update_gradient"))

    def attack(self, data, label, outer_progress_bar, **kwargs):
        self.build_before_attack()
        data = data.cuda()
        delta = torch.zeros_like(data).cuda()
        if self.use_tqdm:
            from tqdm import tqdm
            p_bar = tqdm(range(self.max_iterations))
        for i in (p_bar if self.use_tqdm else range(self.max_iterations)):
            delta = delta.clone().detach()
            delta.requires_grad = True
            # Gradient get
            grad = self.get_gradient(self.src_model, data, delta, label, iteration=i)

            # Gradient update
            grad = self.update_gradient(grad)
            # Update delta
            delta = torch.clamp(delta + self.alpha * grad.sign(), -self.epsilon, self.epsilon)
            delta = torch.clamp(data + delta, 0, 1) - data

            if((i + 1) % self.tester.test_interval == 0) or (i + 1 == self.max_iterations):
                self.tester.test(data + delta, label, outer_progress_bar, iteration=i)
        return data + delta