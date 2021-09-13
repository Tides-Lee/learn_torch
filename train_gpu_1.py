# -*- coding: utf-8 -*-
# 作者: tides
# 日期: 2021/9/10

from torch.utils.tensorboard.writer import SummaryWriter
import torch.optim.optimizer
import torchvision.datasets
from torch.utils.data import DataLoader
from Models import Tudui
from torch import nn
train_data = torchvision.datasets.CIFAR10(root="./dataset", train=True, transform=torchvision.transforms.ToTensor(), download=True)
test_data = torchvision.datasets.CIFAR10(root="./dataset",train=False, transform=torchvision.transforms.ToTensor(), download=True)

# length
train_data_size = len(train_data)
test_data_size = len(test_data)
print("训练数据集的长度为：{},测试数据集的长度为：{}".format(train_data_size, test_data_size))

# use DataLoader to load train&test set
train_dataLoader = DataLoader(train_data, batch_size=64)
test_dataLoader = DataLoader(test_data, batch_size=64)


# 创建网络模型
tudui = Tudui()
tudui.cuda()

# Loss Function
loss_fn = nn.CrossEntropyLoss()
loss_fn.cuda()

# optimize
learn_rate = 1e-2
optimizer = torch.optim.SGD(tudui.parameters(), lr=learn_rate)

# set neural network parameters

total_train_step = 0
total_test_step = 0
epoch = 10

writer = SummaryWriter("./logs_train")


for i in range(epoch):
    print("--------第 {} 轮训练开始了-------".format(i+1))

    for data in train_dataLoader:
        imgs, target = data
        imgs = imgs.cuda()
        target = target.cuda()
        outputs = tudui(imgs)
        loss = loss_fn(outputs, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step += 1
        if total_train_step % 100 == 0:
            print("训练次数为{}时，loss：{}".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    total_test_loss = 0
    with torch.no_grad():
        for data in test_dataLoader:
            imgs, targets = data
            imgs = imgs.cuda()
            targets = targets.cuda()
            outputs = tudui(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss = total_test_loss + loss.item()

    print("整体测试集上的Loss{}".format(total_test_loss))
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    total_test_step += 1

writer.close()
