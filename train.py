import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import os
import torchvision.models as models
from torch.autograd import Variable
import time
import numpy as np
import pickle
import torch.backends.cudnn as cudnn
import sys


DATA = 'imagenet'
use_cuda = True
resume = False
cur = time.time()
save_path = DATA + '_checkpoint'
batch_size = 128
best_acc = 0

seed = np.random.randint(1,1000)
print('Random seed: ', seed)
torch.manual_seed(seed)

print('start normalize')

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

train_transform = transforms.Compose(
                [
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                transforms.ToTensor(),
                normalize,
                ])

test_transform = transforms.Compose(
                [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
                ])

print('start make train_dataset')
train_dataset = torchvision.datasets.ImageFolder(root='../imagenet/train', transform=train_transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                           num_workers=20, pin_memory=True, drop_last=False)#12739

print('start make test_dataset')
test_dataset = torchvision.datasets.ImageFolder(root='../imagenet/val', transform=test_transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=50, shuffle=False,
                                           num_workers=20, pin_memory=True)


print('start build net model')
net = torchvision.models.resnet50()
dtype = torch.float32

criterion = nn.CrossEntropyLoss()

if resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir(save_path), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(save_path + '/resnet50_ckpt.t7')
    net.load_state_dict(checkpoint['net'])


if use_cuda:
    print('start move to cuda')
    torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = True
    net = torch.nn.DataParallel(net, device_ids=[0,1])
    device = torch.device("cuda:0")
    net.to(device=device, dtype=dtype)
    criterion.to(device=device, dtype=dtype)




optimizer = optim.SGD(
    net.parameters(),
    lr=0.1,
    momentum=0.9,
    weight_decay=0.0001,
    nesterov=True
)

# optimizer = optim.RMSprop(
#     net.parameters(),
#     lr=0.045,
#     alpha=0.9,
#     eps=1e-10,
#     weight_decay=0.00004,
#     momentum=0.9,
#     centered=False
# )

def correct(output, target, topk=(1, )):

    maxk = max(topk)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t().type_as(target)
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []

    for k in topk:
        correct_k = correct[:k].view(-1).float().sum().item()
        res.append(correct_k)

    return res

def train(epoch):
    # global monitor
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct1 = 0
    correct5 = 0
    a = time.time()
    #    pred = []
    for batch_idx, (data, target) in enumerate(train_loader):
        if use_cuda:
            data, target = data.to(device=device, dtype=dtype), target.to(device=device)
        data, target = Variable(data), Variable(target)

        optimizer.zero_grad()
        outputs = net(data)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

        train_loss += loss.data[0] * target.size(0)

        corr = correct(outputs, target, topk=(1, 5))

        correct1 += corr[0]
        correct5 += corr[1]

    print('Train loss: %0.5f,     Train_top1: %0.5f,     Train_top5: %0.5f' % (
    train_loss / len(train_loader.dataset), correct1 / len(train_loader.dataset), correct5 / len(train_loader.dataset)))
    print('This epoch cost %0.2f seconds' % (time.time() - a))
    # monitor['train_loss'].append(train_loss / len(train_loader.dataset))
    # monitor['train_acc'].append(correct / len(train_loader.dataset))


def test(epoch):
    global best_acc
        # monitor
    net.eval()
    test_loss = 0
    correct1 = 0
    correct5 = 0
    a = time.time()

    for batch_idx, (data, target) in enumerate(test_loader):
        if use_cuda:
            data, target = data.to(device=device, dtype=dtype), target.to(device=device)
        data, target = Variable(data), Variable(target)
        with torch.no_grad():
            outputs = net(data)
            loss = criterion(outputs, target)

            test_loss += loss.data[0] * target.size(0)
        corr = correct(outputs, target, topk=(1, 5))

        correct1 += corr[0]
        correct5 += corr[1]

    print('Test loss: %0.5f,     Test_top1: %0.5f,     Test_top5: %0.5f' % (
    test_loss / len(test_loader.dataset), correct1 / len(test_loader.dataset), correct5 / len(test_loader.dataset)))
    print('This epoch cost %0.2f seconds' % (time.time() - a))
    # monitor['val_loss'].append(test_loss / len(test_loader.dataset))
    # monitor['val_acc'].append(correct / len(test_loader.dataset))
    acc = correct5 / len(test_loader.dataset)


    if acc > best_acc:
        print('Saving...')
        state = {
            'net': net.state_dict(),
        }

        if not os.path.isdir(save_path):
            os.mkdir(save_path)
        torch.save(state, save_path + '/resnet50_ckpt.t7')
        best_acc = acc


def main():
    start_epoch = 1
    for epoch in range(start_epoch, start_epoch + 150):
        if epoch == 30 or epoch == 60 or epoch == 90:
           optimizer.param_groups[0]['lr'] *= 0.1
        train(epoch)
        test(epoch)
        print('Learning rate: %f' % optimizer.param_groups[0]['lr'])

def debug():
    # global monitor

    net.train()
    train_loss = 0
    correct1 = 0
    correct5 = 0

    #    pred = []
    for batch_idx, (data, target) in enumerate(train_loader):
        a = time.time()
        if use_cuda:
            data, target = data.to(device=device, dtype=dtype), target.to(device=device)
        data, target = Variable(data), Variable(target)

        optimizer.zero_grad()
        outputs = net(data)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

        tmp_loss= loss.data[0] * target.size(0)

        corr = correct(outputs, target, topk=(1, 5))

        correct1 = corr[0]
        correct5 = corr[1]

        print('Batch #%d: '%batch_idx)
        print('Train loss: %0.5f,     Train_top1: %0.5f,     Train_top5: %0.5f' % (
        tmp_loss / batch_size, correct1 / batch_size, correct5 / batch_size))
        print('This batch cost %0.2f seconds' % (time.time() - a))
    # monitor['train_loss'].append(train_loss / len(train_loader.dataset))
    # monitor['train_acc'].append(correct / len(train_loader.dataset))

main()
