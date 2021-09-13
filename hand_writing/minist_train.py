# -*- coding: utf-8 -*-
# 作者: tides
# 日期: 2021/9/12

import torch
from torch import nn
from torch.nn import functional as F
from torch import optim

import torchvision
from matplotlib import pyplot as plt
from utils import plot_image, plot_curve, one_hot

batch_size = 512
# step 1 load data
train_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST('mnist_data', train=True, download=True,
                                                                      transform=torchvision.transforms.Compose([
                                                                          torchvision.transforms.ToTensor(),
                                                                          torchvision.transforms.Normalize(
                                                                              (0.1307,), (0.3081,))
                                                                      ])),
                                           batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST('mnist_data', train=False, download=True,
                                                                     transform=torchvision.transforms.Compose([
                                                                         torchvision.transforms.ToTensor(),
                                                                         torchvision.transforms.Normalize(
                                                                             (0.1307,), (0.3081,))
                                                                     ])),
                                          batch_size=batch_size, shuffle=False)
