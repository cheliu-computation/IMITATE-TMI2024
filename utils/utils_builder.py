from cgi import test
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import numpy as np
import torchvision
from torchvision.models import resnet as torch_resnet
import torch.nn.functional as F
from torch.nn.functional import normalize
from transformers import AutoModel, AutoTokenizer

### mask scale/mask channel
class IMITATE_MaskScale(torch.nn.Module):
    def __init__(self, mhsa_dim=512, device_id='cpu'):
        super(IMITATE_MaskScale, self).__init__()
        resnet = torchvision.models.resnet50(pretrained=True)
        self.device_id = device_id
        self.encoder = resnet
        self.encoder.fc = nn.Identity()
        
        self.mhsa_dim = mhsa_dim
        # image emebedding ==> FINDINGS
        self.cls_token = nn.Parameter(torch.zeros((1, 1, 256), dtype=torch.float32))
        self.positional_embedding = nn.Parameter((torch.zeros((1, self.mhsa_dim+1, 256), dtype=torch.float32)))

        self.pool = nn.AdaptiveAvgPool2d((16,16)) # unify the size
        self.flatten = nn.Flatten(2,3)

        self.multi_attent = nn.MultiheadAttention(256, 4, batch_first=True)
        self.read_proj = nn.Sequential(
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512, affine=False))
        
        self.diag_proj = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512, affine=False))
        
        self.text_proj = nn.Sequential(
            nn.Linear(768, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512, affine=False))

        url = 'emilyalsentzer/Bio_ClinicalBERT'
        self.lm_model = AutoModel.from_pretrained(
            url, trust_remote_code=True, revision='main')
        self.tokenizer = AutoTokenizer.from_pretrained(
            url, trust_remote_code=True, revision='main')

    def _tokenize(self, text):
        tokenizer_output = self.tokenizer.batch_encode_plus(batch_text_or_text_pairs=text,
                                                        add_special_tokens=True,
                                                        truncation=True,
                                                        max_length=256,
                                                        padding='longest',
                                                        return_tensors='pt')
                                            
        # return tokenizer_output
        return tokenizer_output.input_ids, tokenizer_output.attention_mask

    def compute(self, x, mask_method='scale'):
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        x = self.encoder.relu(x)
        x = self.encoder.maxpool(x)
        fea_1 = self.encoder.layer1(x)
        fea_2 = self.encoder.layer2(fea_1)
        fea_3 = self.encoder.layer3(fea_2)
        fea_4 = self.encoder.layer4(fea_3)

        img_emb =  self.encoder.avgpool(fea_4)
        img_emb = img_emb.view(img_emb.shape[0], img_emb.shape[1])

        # read image, random mask different ratio channels, flatten to 256 length sequence
        if mask_method == 'scale':
            fea = [fea_1,fea_2,fea_3,fea_4]
            mask_scale_idx = torch.randint(0, 4, (1,))
            find_emb = self.flatten(self.pool(fea[mask_scale_idx]))
            # if mask scale, self.mhsa_dim > 2048
            find_emb = F.pad(find_emb, (0, 0, 0, self.mhsa_dim-find_emb.shape[1], 0, 0), 'constant', 0)

        elif mask_method == 'channel':
            unmask_1 = torch.randint(0, 256, (int(256*0.25),))
            unmask_2 = torch.randint(0, 512, (int(512*0.15),))
            unmask_3 = torch.randint(0, 1024, (int(1024*0.1),))
            unmask_4 = torch.randint(0, 2048, (int(2048*0.1),))

            fea_1 = self.flatten(self.pool(fea_1[:, unmask_1, :, :]))
            fea_2 = self.flatten(self.pool(fea_2[:, unmask_2, :, :]))
            fea_3 = self.flatten(self.pool(fea_3[:, unmask_3, :, :]))
            fea_4 = self.flatten(self.pool(fea_4[:, unmask_4, :, :]))

            # concat all sequence features
            find_emb = torch.cat([fea_1, fea_2, fea_3, fea_4], dim=1)
            find_emb = F.pad(find_emb, (0, 0, 0, self.mhsa_dim-find_emb.shape[1], 0, 0), 'constant', 0)

        # add positional embedding and cls token
        find_emb = find_emb + self.positional_embedding[:, 1:, :]

        self.cls_tokens = self.cls_token + self.positional_embedding[:, :1, :]
        self.cls_tokens = self.cls_tokens.expand(find_emb.shape[0], -1, -1) 
        find_emb = torch.cat([find_emb, self.cls_tokens], dim=1)
        # attention operation
        find_emb = self.multi_attent(find_emb, find_emb, find_emb, need_weights=False)[0]+find_emb # output shape is (B, 32, 256)
        find_emb = self.read_proj(find_emb[:, -1, :]) # output shape is (B, 32, 256)

        # img_emb = self.encoder(x)
        # find_emb = None
        
        imp_emb = self.diag_proj(img_emb)

        return img_emb, find_emb, imp_emb

    @torch.no_grad()
    def get_text_emb(self, text):
        input_ids, attention_mask = self._tokenize(text)
        text_emb = self.lm_model(input_ids=input_ids.to(self.device_id),
                                 attention_mask=attention_mask.to(self.device_id)).last_hidden_state
        return text_emb
    
    @torch.no_grad()
    def get_find_imp_emb(self, find, imp, all):
        find_emb = self.get_text_emb(find)[:, 0].contiguous()
        # find_emb = self.text_proj(find_emb)

        imp_emb = self.get_text_emb(imp)[:, 0].contiguous()
        # imp_emb = self.text_proj(imp_emb)

        all_emb = self.get_text_emb(all)[:, 0].contiguous()
        all_emb = self.text_proj(all_emb)

        return find_emb, imp_emb, all_emb
    
    @torch.no_grad()
    def get_img_emb(self, x):
        img_emb = self.encoder(x)
        img_emb = self.diag_proj(img_emb)
        return img_emb
    
    def forward(self, x1, x2, find, imp, mask_method=['channel', 'channel']):
        if len(set(mask_method)) == 2:
            img_emb1_m1, find_emb1_m1, imp_emb1_m1 = self.compute(x1, mask_method=mask_method[0])
            img_emb2_m1, find_emb2_m1, imp_emb2_m1 = self.compute(x2, mask_method=mask_method[0])

            img_emb1_m2, find_emb1_m2, imp_emb1_m2 = self.compute(x1, mask_method=mask_method[1])
            img_emb2_m2, find_emb2_m2, imp_emb2_m2 = self.compute(x2, mask_method=mask_method[1])
        else:
            img_emb1_m1, find_emb1_m1, imp_emb1_m1 = self.compute(x1, mask_method=mask_method[0])
            img_emb2_m1, find_emb2_m1, imp_emb2_m1 = self.compute(x2, mask_method=mask_method[0])

            img_emb1_m2, find_emb1_m2, imp_emb1_m2 = self.compute(x1, mask_method=mask_method[0])
            img_emb2_m2, find_emb2_m2, imp_emb2_m2 = self.compute(x2, mask_method=mask_method[0])

        find_emb = self.get_text_emb(find)[:, 0].contiguous()
        imp_emb = self.get_text_emb(imp)[:, 0].contiguous()

        find_emb_proj = self.text_proj(find_emb)
        imp_emb_proj = self.text_proj(imp_emb)

        return {'img_emb1_m1': img_emb1_m1, 'find_emb1_m1': find_emb1_m1, 'imp_emb1_m1': imp_emb1_m1,
                'img_emb2_m1': img_emb2_m1, 'find_emb2_m1': find_emb2_m1, 'imp_emb2_m1': imp_emb2_m1,
                'img_emb1_m2': img_emb1_m2, 'find_emb1_m2': find_emb1_m2, 'imp_emb1_m2': imp_emb1_m2,
                'img_emb2_m2': img_emb2_m2, 'find_emb2_m2': find_emb2_m2, 'imp_emb2_m2': imp_emb2_m2,
                'find_emb': find_emb, 'imp_emb': imp_emb,
                'find_emb_proj': find_emb_proj, 'imp_emb_proj': imp_emb_proj}