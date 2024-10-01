# package import 
import os
from typing import Type
import torch
import torch.nn.functional as F
import torchvision
import pandas as pd
from torch.utils.data.dataloader import DataLoader
import utils_builder
import math
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler as GradScaler
from tqdm import tqdm
import numpy as np
import torch.nn as nn

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

class IMITATE_trainer_wBert:
    def __init__(self, model,
                optimizer, device, model_name, **args):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.model_name = model_name
        self.loss_type = args['loss']
        self.train_batch_size = args['batch_size']
        self.test_batch_size = args['test_batch_size']
        self.max_epochs = args['max_epochs']
        self.lr_max = args['lr']
        self.num_workers = args['num_workers']
        self.checkpoint_interval = args['checkpoint_interval']
        self.smooth = args['smooth']
        self.prior_ratio = args['ratio']

    def precision_at_k(self, output: torch.Tensor, target: torch.Tensor, top_k=(1,)):
        ''' Compute the accuracy over the k top predictions for the specified values of k'''
        with torch.no_grad():
            maxk = max(top_k)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            res = []
            for k in top_k:
                correct_k = correct[:k].contiguous(
                ).view(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            return res
    
    def clip_loss(self, x, y, prior=None, temperature=1):
        smooth = self.smooth

        x = F.normalize(x, dim=-1)
        y = F.normalize(y, dim=-1)
        sim = torch.einsum('i d, j d -> i j', x, y) * 1 / temperature

        labels = torch.arange(x.shape[0]).to(self.device)
        labels = torch.nn.functional.one_hot(labels, num_classes=-1).to(x.dtype)
        if prior is not None:
            prior = torch.corrcoef(prior)
            prior[prior<0] = 0
            prior.fill_diagonal_(0)
            if smooth == 'gau':
                prior = (1/torch.sqrt(torch.tensor(2*torch.pi))) * torch.exp(-0.5*(torch.square(prior)))
            elif smooth == 'lap':
                prior = 0.5 * torch.exp(-torch.abs(prior))
            elif smooth == 'sigmoid':
                prior = torch.sigmoid(prior)
            else:
                prior = 1 - torch.exp(-(self.prior_ratio) * prior)
            prior = prior.to(x.dtype)
            
            labels += prior * temperature

        loss_t = F.cross_entropy(sim, labels) 
        loss_i = F.cross_entropy(sim.T, labels) 

        # i2t_acc1, i2t_acc5 = self.precision_at_k(
        #     sim, labels, top_k=(1, 5))
        # t2i_acc1, t2i_acc5 = self.precision_at_k(
        #     sim.T, labels, top_k=(1, 5))
        # acc1 = (i2t_acc1 + t2i_acc1) / 2.
        # acc5 = (i2t_acc5 + t2i_acc5) / 2.

        return (loss_t + loss_i)

    # traing process
    def train_w_TextEmb(self, train_dataset):
        
        train_loader = DataLoader(train_dataset, batch_size=self.train_batch_size,
                                  num_workers=self.num_workers, 
                                  drop_last=True, shuffle=False,
                                  sampler=DistributedSampler(train_dataset))

        model_checkpoints_folder = os.path.join('proj/') # your path
        if not os.path.exists(model_checkpoints_folder):
            print('create directory "{}" for save checkpoint!'.format(model_checkpoints_folder))
            print('---------------------------')
            os.mkdir(model_checkpoints_folder)
        else:
            print('directory "{}" existing for save checkpoint!'.format(model_checkpoints_folder))

        
        # automatically resume from checkpoint if it exists
        print('#########################################')
        print('Be patient..., checking checkpoint now...')
        if os.path.exists(model_checkpoints_folder + self.model_name+'_checkpoint.pth'):
            ckpt = torch.load(model_checkpoints_folder + self.model_name+'_checkpoint.pth',
                            map_location='cpu')
            start_epoch = ckpt['epoch']
            self.model.load_state_dict(ckpt['model_state_dict'])
            self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            print('continue training successful!')
        else:
            start_epoch = 0
            print('Start training from 0 epoch')

        print('#########################################')
        print('training start!')

        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=33341, T_mult=2, eta_min=1e-8)
        niter = 1

        skip_scheduler = False
        scaler = GradScaler()

        for epoch_counter in tqdm(range(start_epoch, self.max_epochs+1)):

            epoch_loss = 0
            epoch_loss_clip_read, epoch_loss_clip_diag = 0, 0
            total_acc1_read, total_acc5_read = [], []
            total_acc1_diag, total_acc5_diag = [], []

            for data in tqdm(train_loader):
                ##
                imp = data['text_ori']['IMP']
                find = data['text_ori']['FIND']
                find_existed = data['find_existed']
                no_find_idx = [i for i, x in enumerate(find_existed) if x == 'no']

                img_1 = data['image1'].to(self.device).contiguous()
                img_2 = data['image2'].to(self.device).contiguous()

                self.optimizer.zero_grad()

                with autocast():

                    output_dict = self.model(img_1, img_2, find, imp, mask_method=['channel', 'channel'])

                    img_emb1_m1, read_rep_1_m1, diago_rep_1_m1 = output_dict['img_emb1_m1'], output_dict['find_emb1_m1'], output_dict['imp_emb1_m1']
                    img_rep_2_m1, read_rep_2_m1, diago_rep_2_m1 = output_dict['img_emb2_m1'], output_dict['find_emb2_m1'], output_dict['imp_emb2_m1']

                    img_emb1_m2, read_rep_1_m2, diago_rep_1_m2 = output_dict['img_emb1_m2'], output_dict['find_emb1_m2'], output_dict['imp_emb1_m2']
                    img_rep_2_m2, read_rep_2_m2, diago_rep_2_m2 = output_dict['img_emb2_m2'], output_dict['find_emb2_m2'], output_dict['imp_emb2_m2']

                    find_emb, imp_emb = output_dict['find_emb'], output_dict['imp_emb']
                    find_emb_proj, imp_emb_proj = output_dict['find_emb_proj'], output_dict['imp_emb_proj']

                    if self.loss_type == 'CICL': # CICL

                        loss_clip_diag = self.clip_loss(x=diago_rep_1_m1, y=imp_emb_proj, prior=imp_emb, temperature=0.07)+\
                                        self.clip_loss(x=diago_rep_2_m1, y=imp_emb_proj, prior=imp_emb, temperature=0.07)
                        
                        loss_clip_read = self.clip_loss(x=read_rep_1_m1, y=find_emb_proj, prior=find_emb, temperature=0.07) +\
                                    self.clip_loss(x=read_rep_2_m1, y=find_emb_proj, prior=find_emb, temperature=0.07)

                        loss_clip_img_diag = self.clip_loss(x=diago_rep_1_m2, y=diago_rep_2_m2, prior=imp_emb, temperature=0.2) +\
                                        self.clip_loss(x=diago_rep_2_m2, y=diago_rep_1_m2, prior=imp_emb, temperature=0.2)
                        
                        loss_clip_img_read = self.clip_loss(x=read_rep_1_m2, y=read_rep_2_m2, temperature=0.2) +\
                                        self.clip_loss(x=read_rep_2_m2, y=read_rep_1_m2, temperature=0.2)
                        
                        if epoch_counter <= 30:
                            lamb = 1
                        elif epoch_counter < 50:
                            lamb = 1.3 - (epoch_counter-30)/20
                        else:
                            lamb = 0.3

                        loss_clip_read[no_find_idx] = torch.tensor(0).to(self.device)
                        loss_clip_img_read[no_find_idx] = torch.tensor(0).to(self.device)

                        loss = loss_clip_diag + lamb*loss_clip_read + loss_clip_img_diag + lamb*loss_clip_img_read

                        epoch_loss += loss.item()
                        epoch_loss_clip_read += lamb*loss_clip_read.item()
                        epoch_loss_clip_diag += loss_clip_diag.item()

                    scaler.scale(loss).backward()
                    scaler.step(self.optimizer)
                    scaler.update()

                    if not skip_scheduler:
                        scheduler.step() 
                niter += 1
            
            
            if self.device == 0:
                epoch_iter = (len(train_dataset)//self.train_batch_size)
                print(f'{epoch_counter} epoch loss is {epoch_loss/epoch_iter}!')
                print(f'{epoch_counter} epoch read loss is {epoch_loss_clip_read/epoch_iter}!')
                print(f'{epoch_counter} epoch diag loss is {epoch_loss_clip_diag/epoch_iter}!')

                if epoch_counter % 5 == 0:
                    torch.save(self.model.module.encoder.state_dict(),
                    model_checkpoints_folder + self.model_name+f'_{epoch_counter}_encoder.pth')
                    torch.save(self.model.module.state_dict(),
                    model_checkpoints_folder + self.model_name+f'_{epoch_counter}_ckpt.pth')

        # save final model
        torch.save(self.model.module.encoder.state_dict(),
                    model_checkpoints_folder + self.model_name+'_encoder.pth')

    def save_checkpoints(self, epoch, PATH):

        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()},
            PATH)
