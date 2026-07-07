import torch
import torch.nn as nn
import torch.nn.functional as F
from core.registry import *
from core.attacker.BaseAttacker import BaseAttacker
from torch.nn import Sequential
def dropconnect_forward_pre_hook(module, input):
    """
    Forward pre-hook: Apply DropConnect by masking weights and biases before forward pass.
    """
    # if input[0].shape[-1] > 299 // 16:
    #     return
    verbose_time = False

    if verbose_time:
        import time
        start_time = time.time()
    
    drop_prob = getattr(module, "drop_prob", 0.1)

    if not hasattr(module, 'original_weight'):
        module.original_weight = module.weight.data.clone()
    if module.bias is not None and not hasattr(module, 'original_bias'):
        module.original_bias = module.bias.data.clone()

    if verbose_time:
        print(f"save origin weight: {time.time() - start_time:.6f}s")
        start_time = time.time()

    
    mask_weight = torch.bernoulli((1 - drop_prob) * torch.ones_like(module.weight))
    module.weight.data = module.original_weight * mask_weight

    if verbose_time:
        print(f"mask weight: {time.time() - start_time:.6f}s")
        start_time = time.time()

    if module.bias is not None:
        mask_bias = (torch.rand_like(module.bias) > drop_prob).float()
        module.bias.data = module.original_bias * mask_bias
    if verbose_time:
        print(f"mask bias: {time.time() - start_time:.6f}s")

def dropconnect_forward_hook(module, input, output):
    """
    Forward hook: Restore original weights and biases after forward pass.
    """
    # if input[0].shape[-1] > 299 // 16:
    #     return
    verbose_time = False
    if verbose_time:
        import time
        start_time = time.time()
    if hasattr(module, 'original_weight'):
        module.weight.data = module.original_weight
    if module.bias is not None and hasattr(module, 'original_bias'):
        module.bias.data = module.original_bias
    if verbose_time:
        print(f"restore weight: {time.time() - start_time:.6f}s")

def is_selected_layer(name, module, cfg):
    return is_selected_type_layer(name, module, cfg) and is_selected_position_layer(name, module, cfg)
    
def is_selected_type_layer(name, module, cfg):
    type_list = cfg.get("type_list", ["Normalization", "Linear"])
    if 'Linear' in type_list and isinstance(module, nn.Linear):
        return True
    if 'Conv' in type_list and isinstance(module, nn.Conv2d):
        return True
    if 'Normalization' in type_list and (isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.LayerNorm)):
        return True
    return False

def is_selected_position_layer(name, module, cfg):
    drop_name_list = cfg.get("drop_name_list", ["all"])
    if 'all' in drop_name_list:
        return True
    for n in drop_name_list:
        if n in name:
            return True
    return False

def add_dropconnect_hooks_to_selected_layers(model, drop_prob, cfg):
    """
    Add DropConnect forward pre-hooks and forward hooks to specified layers in the model.
    
    Args:
        model: The model to be modified.
        drop_prob: The drop probability for DropConnect.
    """
    
    num = 0  

    for name, module in model.named_modules():
        if is_selected_layer(name, module, cfg):
            module.drop_prob = drop_prob 
            module.register_forward_pre_hook(dropconnect_forward_pre_hook)
            module.register_forward_hook(dropconnect_forward_hook)
            num += 1
            print(f"DropConnect hooks added to layer {name}")
    print(f"Total {num} DropConnect hooks added.")

@ATTACK_REGISTRY.register_module()
class DropConnect(BaseAttacker):
    def __init__(self, **cfg):
        super(DropConnect, self).__init__(**cfg)
        self.drop_prob = cfg.get("drop_prob", 0.01)
        add_dropconnect_hooks_to_selected_layers(self.src_model, self.drop_prob, cfg)
        print(f"DropConnect hooks added to model.")
