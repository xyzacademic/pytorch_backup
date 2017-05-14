#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 20:39:27 2017

@author: wowjoy
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import os 
from resnet_model import ResNet101 as ResNet
from torch.autograd import Variable
import time


best_acc = 0
start_epoch = 0
DATA = 'cifar100'
resume = False
use_cuda = True
save_path = DATA + '_checkpoint'
transform = transforms.Compose(
        [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ])

if DATA == 'cifar100':
    trainset = torchvision.datasets.CIFAR100(root='cifar100', train=True, download=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=4)
    testset = torchvision.datasets.CIFAR100(root='cifar100', train=False, download=False, transform=transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=50, shuffle=False, num_workers=4)
elif DATA == 'cifar10':
    trainset = torchvision.datasets.CIFAR10(root='cifar10', train=True, download=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=8)
    testset = torchvision.datasets.CIFAR10(root='cifar10', train=False, download=False, transform=transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=8)

    
if resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(save_path + '/ckpt.t7')
    net = checkpoint['net']
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']    
    
    
else:
    if DATA =='cifar100':
        net = ResNet(num_classes=100)
    elif DATA == 'cifar10':
        net = ResNet(num_classes=10)
if use_cuda:
    net.cuda()
#    net = torch.nn.DataParallel(net, device_ids=[0,1,2,3])
    
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(
        net.parameters(), 
        lr = 0.1,
        momentum = 0.9,
        weight_decay=5e-4,
        nesterov=True,
        )

def train(epoch):
    print('\nEpoch: %d' %epoch)
    net.train()
    train_loss = 0
    correct = 0
    a = time.time()
    
    for batch_idx, (data, target) in enumerate(train_loader):
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        
        optimizer.zero_grad()
        outputs = net(data)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.data[0] * target.size(0)
        predicted = outputs.data.max(1)[1]
        correct += predicted.eq(target.data).cpu().sum()
        
    print('Train loss: %0.5f,     Train_accuracy: %0.5f' %(train_loss / len(train_loader.dataset), correct / len(train_loader.dataset)))
    print('This epoch cost %0.2f seconds' %(time.time() - a))
        
def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    a = time.time()
    
    for batch_idx, (data, target) in enumerate(test_loader):
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        outputs = net(data)
        loss = criterion(outputs, target)
        
        test_loss += loss.data[0] * target.size(0)
        predicted = outputs.data.max(1)[1]
        correct += predicted.eq(target.data).cpu().sum()
        
    print('Test loss: %0.5f,     Test_accuracy: %0.5f' %(test_loss / len(test_loader.dataset), correct / len(test_loader.dataset)))
    print('This epoch cost %0.2f seconds' %(time.time() - a))
        
    acc = correct / len(test_loader.dataset)
    if acc > best_acc:
        print('Saving...')
        state = {
                'net': net.modules if use_cuda else net,
                'acc': acc,
                'epoch': epoch,
                }
        
        if not os.path.isdir(save_path):
            os.mkdir(save_path)
        torch.save(state, save_path + '/ckpt.t7')
        best_acc = acc
            
for epoch in range(start_epoch, start_epoch + 200):
    train(epoch)
    test(epoch)