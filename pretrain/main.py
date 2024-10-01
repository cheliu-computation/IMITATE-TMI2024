import sys
sys.path.append("../utils")
import utils_builder
import utils_dataset
from utils_trainer import IMITATE_trainer_wBert
from utils_optimizer import LARS
import yaml
from torch.utils.data.dataloader import DataLoader
import torch
import torchvision
import numpy as np
import pandas as pd
import torch.nn as nn
import os

import tempfile
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import random

os.environ["TOKENIZERS_PARALLELISM"] = "true"

def ddp_main():
    dist.init_process_group("nccl")
    torch.cuda.empty_cache()
    rank = dist.get_rank()

    print(f"Start running basic DDP example on rank {rank}.")
    device_id = rank % torch.cuda.device_count()

    # set up
    config = yaml.load(open("config.yaml", "r"), Loader=yaml.FullLoader)

    torch.manual_seed(42)
    random.seed(0)
    np.random.seed(0)
    # loading data path
    text_path = config['text_path']
    img_path = config['img_path']
 
    # define image-text dataset
    train_dataset = utils_dataset.I_T_dataset(image_path=img_path, csv_path=text_path)
    train_dataset = train_dataset.get_dataset(train_test='train')

    # building model part
    # --------------------
    if config['network']['img_model'] == 'resnet50':
        model = utils_builder.IMITATE_MaskScale(device_id=device_id)
    else:
        raise NotImplementedError
    
    model = model.to(device_id)
    model = DDP(model, device_ids=[device_id], find_unused_parameters=True)

    # --------------------

    # choose optimizer (LARS with large batch, AdamW with small batch)
    # --------------------
    optimizer = torch.optim.AdamW(
        model.parameters(),
        **config['optimizer']['params'],
        betas=(0.9, 0.999)
    )


    # ---------xw-----------
    trainer = IMITATE_trainer_wBert(model=model,
                        optimizer=optimizer,
                        device=rank,
                        model_name=config['wandb_name'],
                        **config['trainer'])
    # --------------------

    # --------------------
    # I_T_P_trainer
    trainer.train_w_TextEmb(train_dataset)

ddp_main()