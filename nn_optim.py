# -*- coding: utf-8 -*-
# 作者: tides
# 日期: 2021/9/10
# -*- coding: utf-8 -*-
# 作者: tides
# 日期: 2021/9/10
import torch
import torchvision.datasets
from torch.utils.data import DataLoader
from Models import *

dataset_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])

dataset = torchvision.datasets.CIFAR10(root="./dataset", train=False, transform=dataset_transform, download=True)
dataloader = DataLoader(dataset, batch_size=1)

tudui = Tudui()
optim = torch.optim.SGD(tudui.parameters(), lr=0.01, )
loss = nn.CrossEntropyLoss()


for data in dataloader:
    img, target = data
    outputs = tudui(img)
    result_loss = loss(outputs, target)
    optim.zero_grad()
    result_loss.backward()
    optim.step()
    print("ok")


