from core.registry import TESTER_REGISTRY, DEFENSE_REGISTRY
from core.tools.utils import *
import torch
import torch.nn as nn
from torchvision import transforms
@TESTER_REGISTRY.register_module()
class BaseTester:
    def __init__(self, **cfg):
        self.cfg = cfg
        self.is_targeted = cfg.get("is_targeted", False)
        self.device = cfg.get("device", "cuda")
        self.test_model_names = cfg.get("test_model_names", [])
        self.defense_method_list = cfg.get("defense_method_list", [])
        self.surrogate_model_name = cfg.get("surrogate_model_name", None)
        self.test_models = []
        self.result = {}
        self.defensers = []

        for test_model_name in self.test_model_names:
            test_model = load_model(test_model_name)
            test_model.to(self.device)
            test_model.eval()
            if test_model_name not in cfg.get("not_normalize_names", []):
                test_model = wrap_model(test_model)
            self.test_models.append(test_model)
            self.result[test_model_name] = 0
        
        for defense_method_cfg in self.defense_method_list:
            defenser = DEFENSE_REGISTRY.build(defense_method_cfg)
            defenser.to(self.device)
            self.defensers.append(defenser)
            self.result[defenser.defenser_name] = 0

        if self.surrogate_model_name is not None:
            if self.surrogate_model_name in self.test_model_names:
                self.surrogate_model = self.test_models[self.test_model_names.index(self.surrogate_model_name)]
            else:
                self.surrogate_model = load_model(self.surrogate_model_name)
                self.surrogate_model.to(self.device)
                self.surrogate_model.eval()
                self.surrogate_model = wrap_model(self.surrogate_model)
        self.result["Avg"] = 0
        self.total = 0

    @torch.no_grad()
    def test(self, origin_adv_data, label, progress_bar, **kwargs):
        if self.is_targeted:
            true_label = label[0].to(self.device)
            label = label[1].to(self.device)
            
        else:
            label = label.to(self.device)
            true_label = label

        for test_model_name, test_model in zip(self.test_model_names, self.test_models):
            if test_model_name in self.cfg.get("need_resize_names", []):
                adv_data = transforms.Resize((224, 224))(origin_adv_data).to(self.device)
            else:
                adv_data = origin_adv_data.to(self.device)
            output = test_model(adv_data)

            if self.is_targeted:
                counts = (output.argmax(dim=1) == label).sum().item()
                # counts = (output.argmax(dim=1) == true_label).sum().item()
            else:
                counts = (output.argmax(dim=1) != label).sum().item()
            
            self.result[test_model_name] += counts
            self.result["Avg"] += counts
        
        adv_data = origin_adv_data.to(self.device)
        for defenser in self.defensers:
            
            output = defenser(adv_data)
            if defenser.output_type == "logits":
                pred = output.argmax(dim=1)
            elif defenser.output_type == "pred":
                pred = output
            elif defenser.output_type == "input":
                pred = (self.surrogate_model(output)).argmax(dim=1)
            else:
                raise NotImplementedError
            if self.is_targeted:
                counts = (pred == label).sum().item()
                # counts = (pred == true_label).sum().item()
            else:
                counts = (pred != label).sum().item()
            self.result[defenser.defenser_name] += counts
            self.result["Avg"] += counts
        self.total += adv_data.shape[0]
        avg_asr = self.result["Avg"]/self.total/(len(self.test_models) + len(self.defensers))
        progress_bar.set_postfix(asr=avg_asr)

    def summary(self, **kwargs):
        
        output_str = "|"
        for test_model_name in self.test_model_names:
            output_str += f"{self.result[test_model_name]/self.total} |"
        for defenser in self.defensers:
            output_str += f"{self.result[defenser.defenser_name]/self.total} |"
        output_str += f"{self.result['Avg']/self.total/(len(self.test_models) + len(self.defensers))} |"
        return output_str, self.result