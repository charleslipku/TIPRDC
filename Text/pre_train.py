import torch
from torch import optim
from torch import nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import os
import visdom

import LSTM
import mydataset
import data_handler

torch.manual_seed(0)
os.environ["CUDA_VISIBLE_DEVICES"] = "2, 3"
vis = visdom.Visdom(env="lstm-pric")

total_epoch = 20
batch_size = 256
lr = 0.0001

input_dir = "/root/DATA/NLP/demog-text-removal-master/data/processed/mention_race/"
input_vocab = input_dir + 'vocab'
with open(input_vocab, 'rb') as f:
    vocab = f.readlines()
    vocab = list(map(lambda s: s.strip(), vocab))
vocab_size = len(vocab)


def collate_fn(batch):
    # batch.sort(key=lambda data: len(data[0]), reverse=True)
    token_lists = [token for token, _ in batch]
    data_length = [len(data) for data in token_lists]
    labels = [label for _, label in batch]
    train_data = pad_sequence(token_lists, batch_first=True, padding_value=0)
    labels = torch.Tensor(labels).long()
    return train_data, labels, data_length


data_train, label_train, data_test, label_test = data_handler.get_data("mention", input_dir)
data_train_loader = DataLoader(mydataset.MyDataset(data_train, label_train), batch_size=batch_size, shuffle=True, num_workers=32, collate_fn=collate_fn)
data_test_loader = DataLoader(mydataset.MyDataset(data_test, label_test), batch_size=batch_size, shuffle=False, num_workers=32, collate_fn=collate_fn)

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
    for i, (sentences, labels, sentences_length) in enumerate(data_train_loader):
        if torch.cuda.is_available():
            sentences = sentences.cuda()
            labels = labels.cuda()

        features = FE(sentences)
        output = CF(features, sentences_length)
        loss = criterion(output, labels)

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
        for i, (sentences, labels, sentences_length) in enumerate(data_test_loader):
            if torch.cuda.is_available():
                sentences = sentences.cuda()
                labels = labels.cuda()
            features = FE(sentences)
            output = CF(features, sentences_length)
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


def get_FE_CF():
    FE = LSTM.FeatureExtractor(vocab_size)
    CF = LSTM.Classifier()
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
        torch.save(FE.module, "Models/FE.pth")
        torch.save(CF.module, "Models/CF.pth")
        # torch.save(FE.module, "Models/pre_train/FE.pth")
        # torch.save(CF.module, "Models/pre_train/CF.pth")
    else:
        torch.save(FE, "Models/FE.pth")
        torch.save(CF, "Models/CF.pth")
        # torch.save(FE, "Models/pre_train/FE.pth")
        # torch.save(CF, "Models/pre_train/CF.pth")

    np.array(train_loss).tofile('Result/train_loss.np')
    np.array(test_loss).tofile('Result/test_loss.np')
    np.array(test_acc).tofile('Result/test_acc.np')
    # np.array(train_loss).tofile('Result/pre_train/train_loss.np')
    # np.array(test_loss).tofile('Result/pre_train/test_loss.np')
    # np.array(test_acc).tofile('Result/pre_train/test_acc.np')
    return FE, CF


if __name__ == '__main__':
    # run get_FE or get_ZFE to get a feature extractor whether or not constrained by DIM info
    FE, CF = get_FE_CF()
