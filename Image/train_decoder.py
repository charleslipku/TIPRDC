import torch
from torch import optim
from torch import nn
import numpy as np
import MS_SSIM


z_global_loss = []
zu_global_loss = []

loss_func = MS_SSIM.MS_SSIM(max_val=1)


def train_zdecoder(encoder, decoder, data_train_loader, current_lr, vis=None):
    encoder.train()
    decoder.train()
    optimizer = optim.Adam(decoder.parameters(), lr=current_lr, weight_decay=1e-4)

    for i, (images, labels) in enumerate(data_train_loader):
        if torch.cuda.is_available():
            images, labels = images.cuda(), labels.cuda()
        optimizer.zero_grad()
        features = encoder(images)
        output = decoder(features)
        output[output < 0] = 0
        output[output > 1] = 1
        loss = 1 - loss_func(output, images)
        loss.backward()
        optimizer.step()

        loss = loss.detach().cpu().item()
        index = len(z_global_loss)
        if vis is not None:
            if index == 0:
                print(loss)
                vis.line(Y=np.array([loss]), X=np.array([index]), win='train_zdecoder_loss', opts={
                    'title': 'z-decoder loss'
                })
            else:
                vis.line(Y=np.array([loss]), X=np.array([index]), win='train_zdecoder_loss', update='append')
        z_global_loss.append(loss)

    return decoder


def train_zudecoder(encoder, decoder, data_train_loader, current_lr, vis=None):
    encoder.train()
    decoder.train()
    optimizer = optim.Adam(decoder.parameters(), lr=current_lr, weight_decay=1e-4)

    for i, (images, labels) in enumerate(data_train_loader):
        if torch.cuda.is_available():
            images, labels = images.cuda(), labels.cuda()
        optimizer.zero_grad()
        features = encoder(images)

        new_labels = torch.zeros((len(labels), 2))
        for counter in range(len(labels)):
            if labels[counter] == 0:
                new_labels[counter][0] = 1
            else:
                new_labels[counter][1] = 1

        output = decoder(features, new_labels)
        output[output < 0] = 0
        output[output > 1] = 1
        loss = 1 - loss_func(output, images)
        loss.backward()
        optimizer.step()

        loss = loss.detach().cpu().item()
        index = len(zu_global_loss)
        if vis is not None:
            if index == 0:
                print(loss)
                vis.line(Y=np.array([loss]), X=np.array([index]), win='train_zudecoder_loss', opts={
                    'title': 'zu-decoder loss'
                })
            else:
                vis.line(Y=np.array([loss]), X=np.array([index]), win='train_zudecoder_loss', update='append')
        zu_global_loss.append(loss)

    return decoder
