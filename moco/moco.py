# -*- coding: utf-8 -*-
# @Time : 2022/12/21 11:32
# @Author : KKKc
# @FileName: moco.py
import glob
import math
import PIL.Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets
from torchvision import datasets
from torchvision.transforms.functional import to_pil_image
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18
import torchvision.transforms as transforms
from torchsummary import summary
import numpy as np
from PIL import Image
import os
import time
from collections import OrderedDict
import copy
import random
import matplotlib.pyplot as plt
import cv2 as cv
from PIL import Image
import torchvision.utils as vutils
import common

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        n_feats = 64
        n_resblocks = 3
        E1=[nn.Conv2d(3, n_feats, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, True)]
        E2=[
            common.ResBlock(
                common.default_conv, n_feats, kernel_size=3
            ) for _ in range(n_resblocks)
        ]
        E3=[
            nn.Conv2d(n_feats, n_feats * 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(n_feats * 2, n_feats * 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(n_feats * 2, n_feats * 4, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, True),
            nn.AdaptiveAvgPool2d(1),
        ]
        E=E1+E2+E3
        self.E = nn.Sequential(
            *E
        )
        self.mlp = nn.Sequential(
            nn.Linear(n_feats * 4, n_feats * 4),
            nn.LeakyReLU(0.1, True),
            nn.Linear(n_feats * 4, n_feats * 4),
            nn.LeakyReLU(0.1, True)
        )

        # compress
        self.compress = nn.Sequential(
            nn.Linear(n_feats * 4, n_feats),
            nn.LeakyReLU(0.1, True)
        )

    def forward(self, x):
        fea = self.E(x).squeeze(-1).squeeze(-1)
        fea1 = self.mlp(fea)
        out = self.compress(fea1)
        return out

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def setup_seed(seed):
   torch.manual_seed(seed)
   torch.cuda.manual_seed_all(seed)
   np.random.seed(seed)
   random.seed(seed)
   torch.backends.cudnn.deterministic = True

# 
#setup_seed(20)

# transformation for query and key
train_transform = transforms.Compose([
    transforms.RandomCrop(240),
    transforms.RandomHorizontalFlip(p=0.8),  # 
    transforms.RandomVerticalFlip(p=0.5),  # 
    transforms.ToTensor(),
    # transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])


# define class to make q and k
class Split:
    def __init__(self, transform):
        self.transfrom = transform

    def __call__(self, x):
        q = self.transfrom(x)
        k = self.transfrom(x)
        return [q, k]


# data path
#data_path = '/data/chens/DIV2K_train_HR-160sub/'
#data_path = '/data/chens/DF2K/LR/LRblur/x4/'
data_path =  "/data0/cs/datasets/train/reds_all/" #"/data0/cs/datasets/train/reds_hr_sub/" #'/data0/cs/datasets/train/res_sub_all/' #"/data/cs/DIF2K_subx4_noiselevel_15/LRblur/x4/"  #"/data/cs/DIF2K_rd/"
#data_path = '/data/chens/DIV2K_train_HR-160sub/'
#os.makedirs(data_path, exist_ok=True)


# dataset = torchvision.datasets.ImageFolder('data/BSD200_x4',transform=train_transform)
class MyDataset(Dataset):
    def __init__(self, filename, transform):
        self.filename = filename
        self.transform = transform

    def __len__(self):
        return len(self.filename)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        img_name = self.filename[index]  # 
        img = Image.open(img_name).convert('RGB')
        img1 = self.transform(img)
        img2 = self.transform(img)
        return img1, img2


train_names = sorted(glob.glob(data_path + '*.png', recursive=True))
#np.random.shuffle(train_names)
print(train_names)
print(len(train_names))
dataset = MyDataset(train_names, train_transform)

# dataloader
train_dataloader = DataLoader(dataset, batch_size=64, shuffle=True,num_workers=20,pin_memory=True)
for image1,image2 in train_dataloader:
    print(image1.shape)



q_encoder = Encoder()
k_encoder = copy.deepcopy(q_encoder)

q_encoder = q_encoder.to(device)
k_encoder = k_encoder.to(device)


# summary(q_encoder, (3, 64, 64), device=device.type)


# define loss function
def loss_func(q, k, queue, t=0.07):
    # t: temperature

    N = q.shape[0]  # batch_size
    C = q.shape[1]  # channel

    # bmm: batch matrix multiplication
    pos = torch.exp(torch.div(torch.bmm(q.view(N, 1, C), k.view(N, C, 1)).view(N, 1), t))
    neg = torch.sum(torch.exp(torch.div(torch.mm(q.view(N, C), torch.t(queue)), t)), dim=1)

    # denominator is sum over pos and neg
    denominator = pos + neg

    return torch.mean(-torch.log(torch.div(pos, denominator)))

# define optimizer
opt = optim.Adam(q_encoder.parameters(),lr=3e-4)

# initialize the queue
queue = None
K = 12288 # K: number of negatives to store in queue

# fill the queue with negative samples
flag = 0
if queue is None:
    while True:

        with torch.no_grad():
            for img1, img2 in train_dataloader:
                # extract key samples
                xk = img2.to(device)
                k = k_encoder(xk).detach()

                if queue is None:
                    queue = k
                else:
                    if queue.shape[0] < K: # queue < 8192
                        queue = torch.cat((queue,k),0)
                    else:
                        flag = 1 # stop filling the queue

                if flag == 1:
                    break

        if flag == 1:
            break

queue = queue[:K]
# check queue
print('number of negative samples in queue : ',len(queue))

def adjust_learning_rate(lr,optimizer, epoch, epochs):
    """Decay the learning rate based on schedule"""
    lr = lr
     # cosine lr schedule
    lr *= 0.5 * (1. + math.cos(math.pi * epoch / epochs))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# define function to training
def Training(q_encoder, k_encoder, num_epochs, queue=queue, loss_func=loss_func, opt=opt, data_dl=train_dataloader,
             sanity_check=False):
    loss_history = []
    momentum = 0.999
    start_time = time.time()
    #path2weights = '/content/drive/MyDrive/models/new_q_weights.pt'
    len_data = len(data_dl.dataset)
    # q_encoder.load_state_dict("/content/drive/MyDrive/models/q_weights.pt")
    q_encoder.train()
    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        epoch = epoch 
        q_encoder.train()
        adjust_learning_rate(lr=3e-4,optimizer=opt,epoch=(epoch % 1000),epochs=1000)
        running_loss = 0
        torch.cuda.empty_cache()
        for img1, img2 in data_dl:
            # retrieve query and key
            xq = img1.to(device,non_blocking=True)
            xk = img2.to(device,non_blocking=True)

            # get model outputs
            q = q_encoder(xq)
            k = k_encoder(xk).detach()

            # normalize representations
            q = torch.div(q, torch.norm(q, dim=1).reshape(-1, 1))
            k = torch.div(k, torch.norm(k, dim=1).reshape(-1, 1))

            # get loss value
            loss = loss_func(q, k, queue)
            running_loss += loss
            print("running_loss:{}, learning_rate:{}".format(running_loss,opt.param_groups[0]['lr']))
            opt.zero_grad()
            loss.backward()
            opt.step()

            # update the queue
            queue = torch.cat((queue, k), 0)

            if queue.shape[0] > K:
                queue = queue[256:, :]

            # update k_encoder
            for q_params, k_params in zip(q_encoder.parameters(), k_encoder.parameters()):
                k_params.data.copy_(momentum * k_params + q_params * (1.0 - momentum))

        # store loss history
        epoch_loss = running_loss / len(data_dl.dataset)
        loss_history.append(epoch_loss)
        print('train loss: %.6f, time: %.4f min' % (epoch_loss, (time.time() - start_time) / 60))

        if sanity_check:
            break

        # save weights
        if (epoch+1) % 10 == 0:
          torch.save(q_encoder.state_dict(), '/data0/cs/Encoder/REDS_HR/f64_n_q_weights_{}_{}.pt'.format(epoch+1,epoch_loss))
          #torch.save(k_encoder.state_dict(), '/home/xdu_temp/cs/hr/f256_n_k_weights_{}_{}.pt'.format(epoch+1,epoch_loss))

    return q_encoder, k_encoder, loss_history


# create folder to save q_encoder model weights
#os.makedirs('/home/xdu/cs/ikc-model-hr', exist_ok=True)

# start training
num_epochs = 1000
print("num_epochs",num_epochs)
q_encoder, _, loss_history = Training(q_encoder, k_encoder, num_epochs=num_epochs, sanity_check=False)