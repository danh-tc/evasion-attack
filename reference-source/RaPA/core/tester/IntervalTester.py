from core.registry import TESTER_REGISTRY
from core.tools.utils import *
import torch
import torch.nn as nn
from torchvision import transforms
from .BaseTester import BaseTester
from torchvision.utils import save_image
from core.tools.utils import load_model
import pandas as pd

@TESTER_REGISTRY.register_module()
class IntervalTester(BaseTester):
    def __init__(self, **cfg):
        super(IntervalTester, self).__init__(**cfg)
        self.max_iter = cfg.get("max_iter", 300)
        self.test_interval = cfg.get("test_interval", self.max_iter)
        self.save_dir = cfg.get("save_dir", None)
        self.results_data = []
    @torch.no_grad()
    def test(self, origin_adv_data, origin_label, progress_bar, iteration, **kwargs):
        if self.is_targeted:
            true_label = origin_label[0].to(self.device)
            label = origin_label[1].to(self.device)
            
        else:
            label = origin_label.to(self.device)
            true_label = origin_label
        origin_adv_data = origin_adv_data.to(self.device)

        for test_model_name, test_model in zip(self.test_model_names, self.test_models):
            if test_model_name in self.cfg.get("need_resize_names", []):
                adv_data = transforms.Resize((224, 224))(origin_adv_data)
            else:
                adv_data = origin_adv_data
            output = test_model(adv_data)

            if self.is_targeted:
                counts = (output.argmax(dim=1) == label).sum().item()
            else:
                counts = (output.argmax(dim=1) != label).sum().item()
            
            self.results_data.append([iteration + 1, test_model_name, counts])

        if iteration + 1 == self.max_iter:
            super().test(origin_adv_data, origin_label, progress_bar)

    def summary(self):
        # Create a DataFrame from the accumulated results
        df = pd.DataFrame(self.results_data, columns=['Iteration', 'Model', 'Counts'])

        # Use pivot_table to handle duplicates with an aggregation function
        df_pivot = df.pivot_table(index='Iteration', columns='Model', values='Counts', aggfunc='sum')
        
        os.makedirs(self.save_dir, exist_ok=True)
        # Save the pivoted DataFrame to CSV
        df_pivot.to_csv(f"{self.save_dir}/test_summary.csv")
        
        return super().summary()
        
        