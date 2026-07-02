from core.registry import GRADIENT_REGISTRY
@GRADIENT_REGISTRY.register_module()
class BaseUpdateGradient:
    def __init__(self, **cfg):
        self.cfg = cfg

    def __call__(self, grad):
        grad = grad
        return grad
        