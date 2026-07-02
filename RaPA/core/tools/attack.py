import argparse
import logging
import os
import datetime
from mmengine.config import Config, DictAction
import datetime
import random
import numpy as np
import torch
from tqdm import tqdm
from utils import save_images, set_seed, load_model, wrap_model
from mmengine.registry import Registry
from torch.utils.data import DataLoader
import sys
from core.registry import *
# from dataset import *


def parse_args():
    parser = argparse.ArgumentParser(description='Train with mmcv config')
    parser.add_argument('--config', help='Path to config file')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    return parser.parse_args()

def setup_logger(name, log_file=None, log_level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)  
        file_handler = logging.FileHandler(log_file)
        file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger

def main():
    # 0.1 Config
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)
    cfg.work_dir = os.path.join(cfg.work_dir, datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    os.makedirs(cfg.work_dir, exist_ok=True)
    log_file = os.path.join(cfg.work_dir, cfg.name + '.log') if hasattr(cfg, 'name') else os.path.join(cfg.work_dir, 'train.log')
    logger = setup_logger(name="train_logger", log_file=log_file)
    logger.info('Loaded configuration:')
    logger.info('\n%s', cfg.pretty_text)
    
    assert((cfg.save_images or cfg.test_when_generate) == True), "Please save images or test when generate to get the result"

    # 0.2 Set seed
    set_seed(cfg.seed)

    
    
    # 1 Prepare Generate
    # 1.0 Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1.1 Model
    source_model_cfg = cfg.source_models_cfg
    for source_model_name in source_model_cfg.source_model_names:
        source_model = load_model(source_model_name)

        source_model = source_model.to(device).eval()
        if source_model != 'vit_base_patch16_224':
            source_model = wrap_model(source_model)
        
        # 1.2 Data
        # 1.2.1 Dataset
        dataset_cfg = cfg.dataset_cfg
        dataset_cfg['output_dir'] = os.path.join(cfg.work_dir, source_model_name, 'adv_images')
        dataset = ADVDATASET_REGISTRY.build(dataset_cfg)
        # 1.2.2 DataLoader
        data_loader_cfg = cfg.data_loader_cfg
        data_loader = DataLoader(dataset, **data_loader_cfg)

        # 1.3 Attacker
        attacker_cfg = cfg.attacker_cfg
        attacker_cfg.src_model = source_model
        attacker_cfg.tester_cfg = cfg.get("tester_cfg") if attacker_cfg.get("tester_cfg") is None else attacker_cfg.get("tester_cfg")
        attacker_cfg.tester_cfg.save_dir = os.path.join(cfg.work_dir, source_model_name)
        attacker = ATTACK_REGISTRY.build(attacker_cfg)

        # 2 Generate Samples
        progress_bar = tqdm(total=len(data_loader), desc="Generating samples")
        # 2.1 Prepare data
        for i, (data, label, filenames) in enumerate(data_loader):
            # 2.2 Generate
            adv_data = attacker.attack(data, label, progress_bar)
            # 2.3 Save images
            if cfg.save_images:
                save_images(os.path.join(cfg.work_dir, source_model_name ,'adv_images'), adv_data, filenames)
            progress_bar.update(1)
        
        progress_bar.close()
        output_str, result = attacker.tester.summary()
        logger.info(f"Src Model: {source_model_name}")
        for key, value in result.items():
            logger.info(f"{key}: {value}")
        logger.info(output_str)
        
           

        
    

if __name__ == '__main__':
    main()
