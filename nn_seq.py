# -*- coding: utf-8 -*-
# 作者: tides
# 日期: 2021/9/10
import torch
import torchvision.datasets
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Sequential, Flatten, Linear
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from Models import *

dataset_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])

dataset = torchvision.datasets.CIFAR10(root="./dataset", train=False, transform=dataset_transform, download=True)
dataloader = DataLoader(dataset, batch_size=1)

tudui = Tudui()

loss = nn.CrossEntropyLoss()
for data in dataloader:
    img, target = data
    outputs = tudui(img)
    result_loss = loss(outputs, target)
    result_loss.backward()
    print("ok")


