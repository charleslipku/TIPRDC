import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import os
import visdom

import train_extractor
import LSTM
import mydataset
import data_handler

torch.manual_seed(0)
os.environ["CUDA_VISIBLE_DEVICES"] = "4, 5, 6, 7"
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


def adjust_learning_rate(epoch, init_lr=0.0001):
    schedule = [12]
    cur_lr = init_lr
    for schedule_epoch in schedule:
        if epoch >= schedule_epoch:
            cur_lr *= 0.1
    return cur_lr


def get_FE():
    FE = torch.load("Models/pre_train/FE.pth")
    CF = torch.load("Models/pre_train/CF.pth")
    MI = LSTM.MutlInfo(vocab_size)
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
            FE, CF, MI = train_extractor.train(FE, CF, MI, data_train_loader, current_lr, vis)
            train_extractor.test_classifier(FE, CF, data_test_loader)
    except KeyboardInterrupt:
        pass
    if torch.cuda.device_count() > 1:
        torch.save(FE.module, "Models/extractor/FE_lambda05.pth")
        torch.save(CF.module, "Models/extractor/FE_CF_lambda05.pth")
        torch.save(MI.module, "Models/extractor/FE_MI_lambda05.pth")
    else:
        torch.save(FE, "Models/extractor/FE_lambda05.pth")
        torch.save(CF, "Models/extractor/FE_CF_lambda05.pth")
        torch.save(MI, "Models/extractor/FE_MI_lambda05.pth")

    np.array(train_extractor.global_loss).tofile('Result/extractor/loss_extractor_lambda05.np')
    np.array(train_extractor.target_loss).tofile('Result/extractor/loss_target_lambda05.np')
    np.array(train_extractor.information).tofile('Result/extractor/loss_information_lambda05.np')
    np.array(train_extractor.cf_test_loss).tofile('Result/extractor/loss_extractor_lambda05_test.np')
    np.array(train_extractor.cf_test_acc).tofile('Result/extractor/acc_extractor_lambda05_test.np')
    return FE


def get_ZFE():
    FE = torch.load("Models/pre_train/FE.pth")
    CF = torch.load("Models/pre_train/CF.pth")
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
            FE, CF = train_extractor.train_Z(FE, CF, data_train_loader, current_lr, vis)
            train_extractor.test_classifier(FE, CF, data_test_loader)
    except KeyboardInterrupt:
        pass
    if torch.cuda.device_count() > 1:
        torch.save(FE.module, "Models/extractor/FE_lambda0.pth")
        torch.save(CF.module, "Models/extractor/FE_CF_lambda0.pth")
    else:
        torch.save(FE, "Models/extractor/FE_lambda0.pth")
        torch.save(CF, "Models/extractor/FE_CF_lambda0.pth")

    np.array(train_extractor.global_loss).tofile('Result/extractor/loss_extractor_lambda0.np')
    np.array(train_extractor.cf_test_loss).tofile('Result/extractor/loss_extractor_lambda0_test.np')
    np.array(train_extractor.cf_test_acc).tofile('Result/extractor/acc_extractor_lambda0_test.np')
    return FE


def get_classifier(FE_path):
    FE = torch.load(FE_path)
    for p in FE.parameters():
        p.requires_grad = False
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
            CF = train_extractor.train_classifier(FE, CF, data_train_loader, current_lr, vis)
            train_extractor.test_classifier(FE, CF, data_test_loader)
    except KeyboardInterrupt:
        pass

    if torch.cuda.device_count() > 1:
        torch.save(CF.module, "Models/mention/Classifier_lambda05.pth")
    else:
        torch.save(CF, "Models/mention/Classifier_lambda05.pth")

    np.array(train_extractor.cf_train_loss).tofile('Result/mention/loss_classifier_lambda05_train.np')
    np.array(train_extractor.cf_test_loss).tofile('Result/mention/loss_classifier_lambda05_test.np')
    np.array(train_extractor.cf_test_acc).tofile('Result/mention/acc_classifier_lambda05_test.np')
    return CF


if __name__ == '__main__':
    # run get_FE or get_ZFE to get a feature extractor constrained by DIM info
    FE = get_FE()
    CF = get_classifier("Models/extractor/FE_lambda05.pth")
