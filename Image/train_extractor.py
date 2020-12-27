import torch
from torch import optim
from torch import nn
import numpy as np
from MutualInformation import info_loss


criterion = nn.CrossEntropyLoss()
global_loss = []
target_loss = []
information = []
cf_train_loss = []
cf_test_loss = []
cf_test_acc = []


def train(FE, CF, MI, data_train_loader, current_lr, vis=None, lambda=0.66):
    FE.train()
    CF.train()
    MI.train()
    FE_optimizer = optim.Adam(FE.parameters(), lr=current_lr, weight_decay=1e-4)
    CF_optimizer = optim.Adam(CF.parameters(), lr=current_lr, weight_decay=1e-4)
    MI_optimizer = optim.Adam(MI.parameters(), lr=current_lr, weight_decay=1e-4)

    loss = 0
    for i, (images, labels) in enumerate(data_train_loader):
        if torch.cuda.is_available():
            images, labels = images.cuda(), labels.cuda()

        x = images
        x_prime = torch.cat((x[1:], x[0].unsqueeze(0)), dim=0)
        u = torch.zeros((len(labels), 4))
        for counter in range(len(labels)):
            u[counter][labels[counter]] = 1
        if torch.cuda.is_available():
            u = u.cuda()

        z = FE(images)
        output = CF(z)
        loss_target = criterion(output, labels)
        loss_jsd = -info_loss(MI, x, z, u, x_prime)
        loss = -lambda * loss_target + (1 - lambda) * loss_jsd

        FE_optimizer.zero_grad()
        loss.backward(retain_graph=True)
        FE_optimizer.step()
        z.detach_()

        CF_optimizer.zero_grad()
        loss_target.backward(retain_graph=True)
        CF_optimizer.step()
        output.detach_()

        MI_optimizer.zero_grad()
        loss_jsd.backward()
        MI_optimizer.step()

        loss = -loss.detach().cpu().item()
        loss_target = loss_target.detach().cpu().item()
        loss_jsd = -loss_jsd.detach().cpu().item()

        del u

        index = len(global_loss)
        if vis is not None:
            if index == 0:
                print(loss)
                vis.line(Y=np.array([loss]), X=np.array([index]), win='train_encoder_loss', opts={
                    'title': 'global loss'
                })
                vis.line(Y=np.array([loss_target]), X=np.array([index]), win='train_target_loss', opts={
                    'title': 'target loss'
                })
                vis.line(Y=np.array([loss_jsd]), X=np.array([index]), win='train_jad_loss', opts={
                    'title': 'jsd loss'
                })
            else:
                vis.line(Y=np.array([loss]), X=np.array([index]), win='train_encoder_loss', update='append')
                vis.line(Y=np.array([loss_target]), X=np.array([index]), win='train_target_loss', update='append')
                vis.line(Y=np.array([loss_jsd]), X=np.array([index]), win='train_jad_loss', update='append')
        global_loss.append(loss)
        target_loss.append(loss_target)
        information.append(loss_jsd)
    print(loss)

    return FE, CF, MI


def train_Z(FE, CF, data_train_loader, current_lr, vis=None):
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
        loss_target = criterion(u, labels)
        loss = -loss_target

        FE_optimizer.zero_grad()
        loss.backward(retain_graph=True)
        FE_optimizer.step()
        z.detach_()

        CF_optimizer.zero_grad()
        loss_target.backward(retain_graph=True)
        CF_optimizer.step()

        loss = loss_target.detach().cpu().item()
        index = len(global_loss)
        if vis is not None:
            if index == 0:
                print(loss)
                vis.line(Y=np.array([loss]), X=np.array([index]), win='train_Zencoder_loss', opts={
                    'title': 'global loss'
                })
            else:
                vis.line(Y=np.array([loss]), X=np.array([index]), win='train_Zencoder_loss', update='append')
        global_loss.append(loss)
    print(loss)

    return FE, CF


def train_classifier(FE, CF, data_train_loader, current_lr, vis=None):
    FE.eval()
    CF.train()
    optimizer = optim.Adam(CF.parameters(), lr=current_lr, weight_decay=1e-4)

    for i, (images, labels) in enumerate(data_train_loader):
        if torch.cuda.is_available():
            images, labels = images.cuda(), labels.cuda()
        optimizer.zero_grad()
        features = FE(images).detach()
        output = CF(features)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        loss = loss.detach().cpu().item()
        index = len(cf_train_loss)
        if vis is not None:
            if index == 0:
                print(loss)
                vis.line(Y=np.array([loss]), X=np.array([index]), win='train_cf_loss', opts={
                    'title': 'train loss'
                })
            else:
                vis.line(Y=np.array([loss]), X=np.array([index]), win='train_cf_loss', update='append')
        cf_train_loss.append(loss)

    return CF


def test_classifier(FE, CF, data_test_loader):
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
    cf_test_loss.append(avg_loss)
    cf_test_acc.append(avg_acc)
