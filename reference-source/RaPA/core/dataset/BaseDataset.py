import torch
import os
import pandas as pd
import sys
from core.registry import *

if 'BaseDataset' not in ADVDATASET_REGISTRY._module_dict:
    @ADVDATASET_REGISTRY.register_module()
    class BaseDataset(torch.utils.data.Dataset):
        def __init__(self, input_dir=None, output_dir=None, is_targeted=False, eval=False, label_file='labels.csv', **kwargs):
            self.is_targeted = is_targeted
            self.data_dir = input_dir
            self.f2l = self.load_labels(label_file)
            self.eval = eval
            if eval:
                self.data_dir = output_dir
                print('=> Eval mode: evaluating on {}'.format(self.data_dir))
            else:
                self.data_dir = os.path.join(self.data_dir, 'images')
                print('=> Train mode: training on {}'.format(self.data_dir))
                print('Save images to {}'.format(output_dir))

        def __len__(self):
            return len(self.f2l.keys())

        def load_labels(self, file_name):
            dev = pd.read_csv(file_name)
            if self.is_targeted:
                f2l = {dev.iloc[i][self.filename_col]: [dev.iloc[i][self.label_col] - 1,
                                                        dev.iloc[i][self.target_col] - 1] for i in range(len(dev))}
            else:
                f2l = {dev.iloc[i][self.filename_col]: dev.iloc[i][self.label_col] - 1
                    for i in range(len(dev))}
            return f2l