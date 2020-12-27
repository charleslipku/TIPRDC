import torch
from torchvision import transforms
from torchvision.datasets.folder import ImageFolder
from torch.utils.data import DataLoader
import numpy as np
import os
import visdom
import matplotlib.pyplot as plt

import SegmentVGG16
import train_extractor
import MutualInformation
import decoder
import train_decoder

torch.manual_seed(0)
os.environ["CUDA_VISIBLE_DEVICES"] = "2, 3"
vis = visdom.Visdom(env="vgg16")


transform = transforms.Compose([
    transforms.ToTensor()
])

total_epoch = 10
batch_size = 128
lr = 0.0001
data_train = ImageFolder('/root/DATA/CelebA/tag2/train', transform=transform)
data_test = ImageFolder('/root/DATA/CelebA/tag2/val', transform=transform)
data_train_loader = DataLoader(data_train, batch_size=batch_size, shuffle=True, num_workers=32)
data_test_loader = DataLoader(data_test, batch_size=batch_size, shuffle=True, num_workers=32)

data2_train = ImageFolder('/root/DATA/CelebA/tag5/train', transform=transform)
data2_test = ImageFolder('/root/DATA/CelebA/tag5/val', transform=transform)
data2_train_loader = DataLoader(data2_train, batch_size=batch_size, shuffle=True, num_workers=32)
data2_test_loader = DataLoader(data2_test, batch_size=1, shuffle=False, num_workers=32)


def adjust_learning_rate(epoch, init_lr=0.0001):
    schedule = [12]
    cur_lr = init_lr
    for schedule_epoch in schedule:
        if epoch >= schedule_epoch:
            cur_lr *= 0.1
    return cur_lr


def adjust_learning_rate_classifier(epoch, init_lr=0.0001):
    schedule = [12]
    cur_lr = init_lr
    for schedule_epoch in schedule:
        if epoch >= schedule_epoch:
            cur_lr *= 0.1
    return cur_lr


def adjust_learning_rate_decoder(epoch, init_lr=0.0001):
    schedule = [10]
    cur_lr = init_lr
    for schedule_epoch in schedule:
        if epoch >= schedule_epoch:
            cur_lr *= 0.1
    return cur_lr


def get_FE():
    FE = torch.load("Models/mix/pre_train/FE.pth")
    CF = torch.load("Models/mix/pre_train/CF.pth")
    MI = MutualInformation.MutlInfo()
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
        torch.save(FE.module, "Models/mix/extractor/FE_lambda10.pth")
        torch.save(CF.module, "Models/mix/extractor/FE_CF_lambda10.pth")
        torch.save(MI.module, "Models/mix/extractor/FE_MI_lambda10.pth")
    else:
        torch.save(FE, "Models/mix/extractor/FE_lambda10.pth")
        torch.save(CF, "Models/mix/extractor/FE_CF_lambda10.pth")
        torch.save(MI, "Models/mix/extractor/FE_MI_lambda10.pth")

    np.array(train_extractor.global_loss).tofile('Result/mix/extractor/loss_extractor_lambda10.np')
    np.array(train_extractor.target_loss).tofile('Result/mix/extractor/loss_target_lambda10.np')
    np.array(train_extractor.information).tofile('Result/mix/extractor/loss_information_lambda10.np')
    np.array(train_extractor.cf_test_loss).tofile('Result/mix/extractor/loss_extractor_lambda10_test.np')
    np.array(train_extractor.cf_test_acc).tofile('Result/mix/extractor/acc_extractor_lambda10_test.np')
    return FE


def get_ZFE():
    FE = torch.load("Models/mix/pre_train/FE.pth")
    CF = torch.load("Models/mix/pre_train/CF.pth")
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
        torch.save(FE.module, "Models/mix/extractor/FE_lambda0.pth")
        torch.save(CF.module, "Models/mix/extractor/FE_CF_lambda0.pth")
    else:
        torch.save(FE, "Models/mix/extractor/FE_lambda0.pth")
        torch.save(CF, "Models/mix/extractor/FE_CF_lambda0.pth")

    np.array(train_extractor.global_loss).tofile('Result/mix/extractor/loss_extractor_lambda0.np')
    np.array(train_extractor.cf_test_loss).tofile('Result/mix/extractor/loss_extractor_lambda0_test.np')
    np.array(train_extractor.cf_test_acc).tofile('Result/mix/extractor/acc_extractor_lambda0_test.np')
    return FE


def get_zdecoder(FE_path):
    FE = torch.load(FE_path)
    for p in FE.parameters():
        p.requires_grad = False
    DC = decoder.ZDecoder()
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            FE = torch.nn.DataParallel(FE)
            DC = torch.nn.DataParallel(DC)
        FE = FE.cuda()
        DC = DC.cuda()
    try:
        for epoch in range(total_epoch):
            print("epoch %d" % epoch)
            cur_lr = adjust_learning_rate_decoder(epoch, lr)
            DC = train_decoder.train_zdecoder(FE, DC, data_train_loader, cur_lr, vis)
    except KeyboardInterrupt:
        pass
    if torch.cuda.device_count() > 1:
        torch.save(DC.module, "Models/gender/decoder/decoder_WithoutLabel_lambda0.pth")
    else:
        torch.save(DC, "Models/gender/decoder/decoder_WithoutLabel_lambda0.pth")

    np.array(train_decoder.z_global_loss).tofile('Result/gender/decoder/loss_WithoutLabel_lambda0.np')
    return DC


def get_classifier(FE_path):
    FE = torch.load(FE_path)
    for p in FE.parameters():
        p.requires_grad = False
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
            current_lr = adjust_learning_rate_classifier(epoch, lr)
            CF = train_extractor.train_classifier(FE, CF, data2_train_loader, current_lr, vis)
            train_extractor.test_classifier(FE, CF, data2_test_loader)
    except KeyboardInterrupt:
        pass

    if torch.cuda.device_count() > 1:
        torch.save(CF.module, "Models/mix/smiling/Classifier_lambda10.pth")
    else:
        torch.save(CF, "Models/mix/smiling/Classifier_lambda10.pth")

    np.array(train_extractor.cf_train_loss).tofile('Result/mix/smiling/loss_classifier_lambda10_train.np')
    np.array(train_extractor.cf_test_loss).tofile('Result/mix/smiling/loss_classifier_lambda10_test.np')
    np.array(train_extractor.cf_test_acc).tofile('Result/mix/smiling/acc_classifier_lambda10_test.np')
    return CF


def show_result(FE_path, DC_path, save_path, data_test_loader, withU=False, image_counter=2):
    FE = torch.load(FE_path)
    DC = torch.load(DC_path)
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            FE = torch.nn.DataParallel(FE)
            DC = torch.nn.DataParallel(DC)
        FE = FE.cuda()
        DC = DC.cuda()
    FE.eval()
    DC.eval()
    for i, (images, labels) in enumerate(data_test_loader):
        if i != image_counter:
            continue
        img = images[0]
        img = img.numpy()
        img = np.transpose(img, (1, 2, 0))
        # plt.imshow(img)
        plt.imsave('images/%d/raw.eps' % image_counter, img, format='eps')

        if withU:
            new_labels = torch.zeros((len(labels), 2))
            for counter in range(len(labels)):
                if labels[counter] == 0:
                    new_labels[counter][0] = 1
                else:
                    new_labels[counter][1] = 1

            if torch.cuda.is_available():
                images, new_labels = images.cuda(), new_labels.cuda()
            reconstruct_imgs = DC(FE(images), new_labels)
        else:
            if torch.cuda.is_available():
                images = images.cuda()
            reconstruct_imgs = DC(FE(images))
        reconstruct_img = reconstruct_imgs[0]
        reconstruct_img[reconstruct_img < 0] = 0
        reconstruct_img[reconstruct_img > 1] = 1
        reconstruct_img = reconstruct_img.cpu().detach().numpy()
        reconstruct_img = np.transpose(reconstruct_img, (1, 2, 0))

        # plt.imshow(reconstruct_img)
        plt.imsave(save_path, reconstruct_img, format='eps')

        break
    plt.show()
    # plt.savefig('Result/img/loss_with_info_constrain_or_not.eps', format='eps')


if __name__ == '__main__':
    # run get_FE or get_ZFE to get a feature extractor whether or not constrained by DIM info
    # FE = get_FE()
    # FE = get_ZFE()
    # ZD = get_zdecoder("Models/gender/extractor/FE_lambda0.pth")
    # CF = get_classifier("Models/mix/extractor/FE_lambda10.pth")

    image_list = [2, 5, 9, 16, 19, 31, 20, 26, 54, 77, 79, 82]
    attribute_list = ['age', 'gender']
    rate_list = ['0', '01', '05', '1', '10']
    for image in image_list:
        if not os.path.exists('images/%d' % image):
            os.makedirs('images/%d' % image)
        for attribute in attribute_list:
            for rate in rate_list:
              show_result("Models/%s/extractor/FE_lambda%s.pth" % (attribute, rate),
                          "Models/%s/decoder/decoder_WithoutLabel_lambda%s.pth" % (attribute, rate),
                          "images/%d/%s_lambda%s.eps" % (image, attribute, rate),
                          data2_test_loader,
                          image_counter=image)
