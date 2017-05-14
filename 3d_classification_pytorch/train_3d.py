#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  3 11:21:35 2017

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
from sklearn.metrics import confusion_matrix, classification_report

train_data = np.load('train_data.npy')
train_label = np.load('train_label.npy')
test_data = np.load('test_data.npy')
test_label = np.load('test_label.npy')

#train_data, train_label, test_data, test_label_ = torch.from_numpy(train_data), torch.from_numpy(train_label),torch.from_numpy(test_data), torch.from_numpy(test_label)
#trainset = torch.utils.data.TensorDataset(train_data, train_label)
#testset = torch.utils.data.TensorDataset(test_data, test_label_)
#train_loader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4)
#test_loader = torch.utils.data.DataLoader(testset, batch_size=50, shuffle=False, num_workers=4)
#
#
#DATA = 'resnet3d_11'
#resume = False
#use_cuda = True
#save_path = DATA + '_checkpoint'
#start_epoch = 1
#if resume:
#    # Load checkpoint.
#    print('==> Resuming from checkpoint..')
#    assert os.path.isdir(save_path), 'Error: no checkpoint directory found!'
#    checkpoint = torch.load(save_path + '/34_ckpt.t7')
#    net = checkpoint['net']
#
#    start_epoch = checkpoint['epoch']   
# 
#else:
#    net = resnet(n=8, k=1, num_classes = 2)
#    net = torch.nn.DataParallel(net, device_ids=[0,1])
#
#if use_cuda:
#    net.cuda()
#    
#criterion = nn.CrossEntropyLoss()
#optimizer = optim.SGD(
#        net.parameters(), 
#        lr = 0.1,
#        momentum = 0.9,
#        weight_decay=5e-4,
#        nesterov=True,
#        )
#
#def train(epoch):
#    print('\nEpoch: %d' %epoch)
#    net.train()
#    train_loss = 0
#    correct = 0
#    a = time.time()
#    
#    for batch_idx, (data, target) in enumerate(train_loader):
#        data = torch.tanh((data+400)/600)
#        if use_cuda:
#            data, target = data.cuda(), target.cuda()
#        data, target = Variable(data), Variable(target)
#        
#        optimizer.zero_grad()
#        outputs = net(data)
#        loss = criterion(outputs, target)
#        loss.backward()
#        optimizer.step()
#        
#        train_loss += loss.data[0] * target.size(0)
#        predicted = outputs.data.max(1)[1]
#        correct += predicted.eq(target.data).cpu().sum()
#        
#    print('Train loss: %0.5f,     Train_accuracy: %0.5f' %(train_loss / len(train_loader.dataset), correct / len(train_loader.dataset)))
#    print('This epoch cost %0.2f seconds' %(time.time() - a))
#    
#def test(epoch):
#    global best_acc
#    net.eval()
#    test_loss = 0
#    correct = 0
#    a = time.time()
#    pred = []
#    for batch_idx, (data, target) in enumerate(test_loader):
#        data = torch.tanh((data+400)/600)
#        if use_cuda:
#            data, target = data.cuda(), target.cuda()
#        data, target = Variable(data), Variable(target)
#        outputs = net(data)
#        loss = criterion(outputs, target)
#        
#        test_loss += loss.data[0] * target.size(0)
#        predicted = outputs.data.max(1)[1]
#        pred += predicted.cpu().numpy().reshape(-1,).tolist()
#        correct += predicted.eq(target.data).cpu().sum()
#        
#    print('Test loss: %0.5f,     Test_accuracy: %0.5f' %(test_loss / len(test_loader.dataset), correct / len(test_loader.dataset)))
#    print('This epoch cost %0.2f seconds' %(time.time() - a))
#    pred = np.array(pred)    
#    target_names = ['False_nodule', 'True_nodule']
#    score = classification_report(test_label, pred, target_names=target_names)
#    print(score)    
#    
#
#    print('Saving...')
#    state = {
#            'net': net,
#
#            'epoch': epoch,
#            }
#        
#    if not os.path.isdir(save_path):
#        os.mkdir(save_path)
#    torch.save(state, save_path + '/'+str(epoch)+'_ckpt.t7')
#
#for epoch in range(start_epoch, start_epoch + 200):
#    if epoch%60 == 0:
#            optimizer.param_groups[0]['lr'] *= 0.1 
#    train(epoch)
#    test(epoch)
#    print('Learning rate: %f' % optimizer.param_groups[0]['lr'])