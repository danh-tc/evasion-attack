from core.registry import TRANSFORM_REGISTRY
@TRANSFORM_REGISTRY.register_module()
class BaseTransform:
    def __init__(self, **cfg):
        self.cfg = cfg
        self.type = cfg.get("type", "BaseTransform")

    def __call__(self, X_in):
        return X_in