import torch
import pandas as pd
from torch.utils.data import Dataset
import numpy as np
from torchvision.transforms import transforms
import cv2
from PIL import Image
import skimage
from tqdm import tqdm
import re

class MIMIC_img_text_embed_dataset(Dataset):
    def __init__(self, image_data, transform=None, **args):
        self.img_data = image_data
        self.text_csv = args['text']
        self.mode = args['train_test']
        self.transform = transform

    def __len__(self):
        return self.text_csv.shape[0]
       
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # extract image
        img = self.img_data[idx]
        img = Image.fromarray(img).convert("RGB")

        findings = self.text_csv['findings'].iloc[idx]
        impression = self.text_csv['impression'].iloc[idx]
        
        if findings == 'dumb' or type(findings) == float:
            findings = str(findings)
            find_existed = 'no'
        else:
            impression = str(impression)
            find_existed = 'yes'

        text = {'FIND': findings, 'IMP': impression, 'ALL': findings+impression}
        
        sample = {'image':img, 'text_ori': text, 'find_existed': find_existed}

        if self.transform:
            if self.mode == 'train':
                sample['image1'] = self.transform[0](sample['image'])
                sample['image2'] = self.transform[1](sample['image'])
            else:
                sample['image1'] = self.transform[0](sample['image'])
                sample['image2'] = self.transform[1](sample['image'])
        # pop the key 'image'
        sample.pop('image')
        
        return sample


class I_T_dataset:

    def __init__(self, image_path, csv_path, database='MIMIC', **args):
        self.image_path = image_path
        self.csv_path = csv_path

    def get_dataset(self, train_test, T=None):
        normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))


        if train_test == 'train':
            print('Apply Train-stage Transform!')

            Transforms = transforms.Compose([
            transforms.RandomCrop(224),
                transforms.RandomGrayscale(p=0.5),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=30),
                transforms.RandomPerspective(distortion_scale=0.5,
                                                p=0.5,
                                                interpolation=3),
                transforms.RandomAutocontrast(p=0.5),
                transforms.ToTensor(),
                normalize
            ])

            Transforms_super = transforms.Compose([
                transforms.RandomCrop(224),
                transforms.RandomGrayscale(p=0.5),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=30),
                transforms.RandomPerspective(distortion_scale=0.5,
                                                p=0.5,
                                                interpolation=3),
                transforms.RandomAutocontrast(p=0.5),
                transforms.ToTensor(),
                normalize
            ])

            Trans = [Transforms, Transforms_super]
        else:
            Transforms = transforms.Compose([
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize
            ])

            Transforms_super = transforms.Compose([
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize
            ])

            Trans = [Transforms, Transforms_super]
        img_path = np.load(
            self.image_path, allow_pickle=True, mmap_mode='r')
        csv_path = pd.read_csv(
            self.csv_path, low_memory=False)
        
        print('Total number of samples: {}'.format(csv_path.shape[0]))

        misc_args = {'train_test': train_test,
                   'text': csv_path}

        dataset = MIMIC_img_text_embed_dataset(image_data=img_path,
                                       transform=Trans,
                                       **misc_args)

        return dataset
