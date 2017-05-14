#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 11:15:58 2017

@author: wowjoy
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class Block(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(Block, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(planes),
                    )
            
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
    
    
class Resnet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, k=1):
        super(Resnet, self).__init__()
        self.in_planes=16
        self.k = k
        self.bn0 = nn.BatchNorm2d(3)
        self.conv1 = conv3x3(3, 16)
        self.bn1 = nn.BatchNorm2d(16)
        
        self.layer1 = self._make_layer(block, 16*k, num_blocks, stride=1)
        self.layer2 = self._make_layer(block, 32*k, num_blocks, stride=2)
        self.layer3 = self._make_layer(block, 64*k, num_blocks, stride=2)
        self.linear = nn.Linear(64*k, num_classes)
 
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.bn0(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, 8, 1)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
    
def resnet(n, k, num_classes):
    return Resnet(Block, num_blocks=n, k = k, num_classes=num_classes)


def test_resnet():
    net = resnet(3,1,10)
    y = net(Variable(torch.rand(1,3,32,32)))
    print(y.size())
        