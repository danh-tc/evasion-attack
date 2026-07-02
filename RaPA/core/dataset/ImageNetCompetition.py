import torch
import os
import pandas as pd
from PIL import Image
from torchvision import transforms
from .BaseDataset import BaseDataset
from core.registry import *

@ADVDATASET_REGISTRY.register_module()
class ImageNetCompetition(BaseDataset):
    def __init__(self, input_dir=None, output_dir=None, is_targeted=False, eval=False, label_file='./dataset/images.csv'):
        self.filename_col = 'ImageId'
        self.label_col = 'TrueLabel'
        self.target_col = 'TargetClass'
        super().__init__(input_dir, output_dir, is_targeted, eval, label_file)

    def __getitem__(self, idx):
        filename = list(self.f2l.keys())[idx]
        assert isinstance(filename, str)
        filepath = os.path.join(self.data_dir, filename + '.png')
        if not os.path.exists(filepath):
            filepath = os.path.join(self.data_dir, filename + '.JPEG')
            print("Pay attention to the file extension, the JPEG will compress the image")
        image = Image.open(filepath)
        trn = transforms.Compose([transforms.ToTensor(), ])
        image = trn(image)
        label = self.f2l[filename]
        return image, label, filename
    