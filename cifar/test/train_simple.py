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
from resnet_simple import resnet
from torch.autograd import Variable
import time
import numpy as np

train_data = np.load('cifar10_traindata.npy').astype(np.float32)
train_label = np.load('cifar10_trainlabel.npy')
test_data = np.load('cifar10_testdata.npy').astype(np.float32)
test_label = np.load('cifar10_testlabel.npy')




train_data, train_label, test_data, test_label = torch.from_numpy(train_data), torch.from_numpy(train_label),torch.from_numpy(test_data), torch.from_numpy(test_label)
trainset = torch.utils.data.TensorDataset(train_data, train_label)
testset = torch.utils.data.TensorDataset(test_data, test_label)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4)
test_loader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=4)

best_acc = 0
start_epoch = 1
DATA = 'cifar10'
resume = True
use_cuda = True
save_path = DATA + '_checkpoint'
#transform = transforms.Compose(
#        [
#                
#                transforms.RandomCrop(32, padding=4),
#                transforms.RandomHorizontalFlip(),
#                transforms.ToTensor(),
#                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
#                ])
#
#if DATA == 'cifar100':
#    trainset = torchvision.datasets.CIFAR100(root='cifar100', train=True, download=False, transform=transform)
#    train_loader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4)
#    testset = torchvision.datasets.CIFAR100(root='cifar100', train=False, download=False, transform=transform)
#    test_loader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=4)
#elif DATA == 'cifar10':
#    trainset = torchvision.datasets.CIFAR10(root='cifar10', train=True, download=False, transform=transform)
#    train_loader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=8)
#    testset = torchvision.datasets.CIFAR10(root='cifar10', train=False, download=False, transform=transform)
#    test_loader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=8)
#
#    
if resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir(save_path), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(save_path + '/params.pkl')
    net = resnet(n=3, k=1, num_classes = 10)
    if use_cuda:
        net = net.cuda()    
        net = torch.nn.DataParallel(net, device_ids=[0,1] )
    net.load_state_dict(checkpoint)


#    net = checkpoint['net']
#    best_acc = checkpoint['acc']
#    start_epoch = checkpoint['epoch']    
    
    
else:
    if DATA =='cifar100':
        net = resnet(n=3, k=1, num_classes = 100)
    elif DATA == 'cifar10':
        net = resnet(n=3, k=1, num_classes = 10)
        
    if use_cuda:
        net.cuda()
        net = torch.nn.DataParallel(net, device_ids=[0,1])
        
    
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
#        state = {
##                'net': net.modules if use_cuda else net,
#                'params': net.state_dict(),
#                'acc': acc,
#                'epoch': epoch,
#                }
#        
        state = net.state_dict()
        if not os.path.isdir(save_path):
            os.mkdir(save_path)
        torch.save(state, save_path + '/params.pkl')
        best_acc = acc
            
for epoch in range(start_epoch, start_epoch + 200):
    if epoch%60 == 0:
            optimizer.param_groups[0]['lr'] *= 0.1 
    train(epoch)
    test(epoch)
    print('Learning rate: %f' % optimizer.param_groups[0]['lr'])