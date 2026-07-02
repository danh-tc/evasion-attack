from mmengine.registry import Registry
ATTACK_REGISTRY = Registry('attack_method')  
TRANSFORM_REGISTRY = Registry('input_transform')  
LOSS_FN_REGISTRY = Registry('loss_fn')  
GRADIENT_REGISTRY = Registry('gradient')  
ADVDATASET_REGISTRY = Registry('adv_dataset')
TESTER_REGISTRY = Registry('tester')
DEFENSE_REGISTRY = Registry('defense_method')
