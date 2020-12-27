import torch
from torch import optim
from torch import nn
from torchvision import transforms
from torchvision.datasets.folder import ImageFolder
from torch.utils.data import DataLoader
import numpy as np
import os
import visdom

import SegmentVGG16
from MutualInformation import info_loss
from MutualInformation import MutlInfo

torch.manual_seed(0)
os.environ["CUDA_VISIBLE_DEVICES"] = "3, 4, 5, 6, 7"
vis = visdom.Visdom(env="vgg16")



transform = transforms.Compose([
    transforms.ToTensor()
])

total_epoch = 20
batch_size = 256
lr = 0.0001
data_train = ImageFolder('/root/DATA/CelebA/tag7/train', transform=transform)
data_test = ImageFolder('/root/DATA/CelebA/tag7/val', transform=transform)
data_train_loader = DataLoader(data_train, batch_size=batch_size, shuffle=True, num_workers=32)
data_test_loader = DataLoader(data_test, batch_size=batch_size, shuffle=False, num_workers=32)

criterion = nn.CrossEntropyLoss()
train_loss = []
test_loss = []
test_acc = []


def adjust_learning_rate(epoch, init_lr=0.0001):
    schedule = [12]
    cur_lr = init_lr
    for schedule_epoch in schedule:
        if epoch >= schedule_epoch:
            cur_lr *= 0.1
    return cur_lr


def train_FE_CF(FE, CF, data_train_loader, current_lr, vis=None):
    FE.train()
    CF.train()
    FE_optimizer = optim.Adam(FE.parameters(), lr=current_lr, weight_decay=1e-4)
    CF_optimizer = optim.Adam(CF.parameters(), lr=current_lr, weight_decay=1e-4)

    loss = 0
    for i, (images, labels) in enumerate(data_train_loader):
        if torch.cuda.is_available():
            images, labels = images.cuda(), labels.cuda()

        z = FE(images)
        u = CF(z)
        loss = criterion(u, labels)

        FE_optimizer.zero_grad()
        CF_optimizer.zero_grad()
        loss.backward()
        CF_optimizer.step()
        FE_optimizer.step()

        index = len(train_loss)
        loss = loss.detach().cpu().item()
        if vis is not None:
            if index == 0:
                print(loss)
                vis.line(Y=np.array([loss]), X=np.array([index]), win='pretrain_train_loss', opts={
                    'title': 'train loss'
                })
            else:
                vis.line(Y=np.array([loss]), X=np.array([index]), win='pretrain_train_loss', update='append')
        train_loss.append(loss)
    print(loss)

    return FE, CF


def test_FE_CF(FE, CF, data_test_loader):
    FE.eval()
    CF.eval()

    avg_loss = 0
    avg_acc = 0
    counter = 0

    with torch.no_grad():
        for i, (images, labels) in enumerate(data_test_loader):
            if torch.cuda.is_available():
                images, labels = images.cuda(), labels.cuda()
            features = FE(images)
            output = CF(features)
            avg_loss += criterion(output, labels).sum()
            pred = output.detach().max(1)[1]
            avg_acc += pred.eq(labels.view_as(pred)).sum()
            counter += 1

    avg_loss /= counter
    avg_loss = avg_loss.detach().cpu().item()
    avg_acc = float(avg_acc) / len(data_test_loader)
    print('Test Avg. Loss: %f, Accuracy: %f' % (avg_loss, avg_acc))
    test_loss.append(avg_loss)
    test_acc.append(avg_acc)


def train_MI(FE, CF, MI, data_train_loader, current_lr, vis=None):
    FE.train()
    CF.train()
    MI.train()
    MI_optimizer = optim.Adam(MI.parameters(), lr=current_lr, weight_decay=1e-4)

    loss = 0
    for i, (images, labels) in enumerate(data_train_loader):
        if torch.cuda.is_available():
            images, labels = images.cuda(), labels.cuda()
        x = images
        x_prime = torch.cat((x[1:], x[0].unsqueeze(0)), dim=0)
        z = FE(images).detach()
        u = CF(z).detach()
        loss = -info_loss(MI, x, z, u, x_prime)
        MI_optimizer.zero_grad()
        loss.backward()
        MI_optimizer.step()

        index = len(train_loss)
        loss = -loss.detach().cpu().item()
        if vis is not None:
            if index == 0:
                print(loss)
                vis.line(Y=np.array([loss]), X=np.array([index]), win='pretrain_train_mi', opts={
                    'title': 'MI'
                })
            else:
                vis.line(Y=np.array([loss]), X=np.array([index]), win='pretrain_train_mi', update='append')
        train_loss.append(loss)
    print(loss)

    return MI


def test_MI(FE, CF, MI, data_test_loader):
    FE.eval()
    CF.eval()
    MI.eval()

    avg_loss = 0
    counter = 0

    with torch.no_grad():
        for i, (images, labels) in enumerate(data_test_loader):
            if torch.cuda.is_available():
                images = images.cuda(), labels.cuda()
            x = images
            x_prime = torch.cat((x[1:], x[0].unsqueeze(0)), dim=0)
            z = FE(images).detach()
            u = CF(z).detach()
            avg_loss += info_loss(MI, x, z, u, x_prime)
            counter += 1

    avg_loss /= counter
    avg_loss = avg_loss.detach().cpu().item()
    print('Test Avg. Loss: %f' % avg_loss)
    test_loss.append(avg_loss)


def get_FE_CF():
    FE = SegmentVGG16.FeatureExtractor()
    CF = SegmentVGG16.Classifier()
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            FE = torch.nn.DataParallel(FE)
            CF = torch.nn.DataParallel(CF)
        FE = FE.cuda()
        CF = CF.cuda()
    try:
        for epoch in range(total_epoch):
            print("epoch %d" % epoch)
            current_lr = adjust_learning_rate(epoch, lr)
            FE, CF = train_FE_CF(FE, CF, data_train_loader, current_lr, vis)
            test_FE_CF(FE, CF, data_test_loader)
    except KeyboardInterrupt:
        pass
    if torch.cuda.device_count() > 1:
        torch.save(FE.module, "Models/mix/pre_train/FE.pth")
        torch.save(CF.module, "Models/mix/pre_train/CF.pth")
        # torch.save(FE.module, "Models/pre_train/FE.pth")
        # torch.save(CF.module, "Models/pre_train/CF.pth")
    else:
        torch.save(FE, "Models/mix/pre_train/FE.pth")
        torch.save(CF, "Models/mix/pre_train/CF.pth")
        # torch.save(FE, "Models/pre_train/FE.pth")
        # torch.save(CF, "Models/pre_train/CF.pth")

    np.array(train_loss).tofile('Result/mix/pre_train/train_loss.np')
    np.array(test_loss).tofile('Result/mix/pre_train/test_loss.np')
    np.array(test_acc).tofile('Result/mix/pre_train/test_acc.np')
    # np.array(train_loss).tofile('Result/pre_train/train_loss.np')
    # np.array(test_loss).tofile('Result/pre_train/test_loss.np')
    # np.array(test_acc).tofile('Result/pre_train/test_acc.np')
    return FE, CF


def get_MI():
    FE = torch.load("Models/pre_train/FE.pth")
    CF = torch.load("Models/pre_train/CF.pth")
    MI = MutlInfo()
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            FE = torch.nn.DataParallel(FE)
            CF = torch.nn.DataParallel(CF)
            MI = torch.nn.DataParallel(MI)
        FE = FE.cuda()
        CF = CF.cuda()
        MI = MI.cuda()
    try:
        for epoch in range(total_epoch):
            print("epoch %d" % epoch)
            current_lr = adjust_learning_rate(epoch, lr)
            MI = train_MI(FE, CF, MI, data_test_loader, current_lr, vis)
            test_MI(FE, CF, MI, data_test_loader)
    except KeyboardInterrupt:
        pass
    if torch.cuda.device_count() > 1:
        torch.save(MI.module, "Models/pre_train/MI.pth")
    else:
        torch.save(MI, "Models/pre_train/MI.pth")

    np.array(train_loss).tofile('Result/pre_train/train_mi.np')
    np.array(test_loss).tofile('Result/pre_train/test_mi.np')
    return MI


if __name__ == '__main__':
    # run get_FE or get_ZFE to get a feature extractor whether or not constrained by DIM info
    FE, CF = get_FE_CF()
    # MI = get_MI()



