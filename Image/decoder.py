import torch
import torch.nn as nn


class ZDecoder(nn.Module):
    def __init__(self):
        super(ZDecoder, self).__init__()

        self.convTrans = nn.Sequential(
            nn.Upsample(size=(109, 89), mode='bilinear'),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128, 64, kernel_size=3, padding=1),

            nn.Upsample(size=(218, 178), mode='bilinear'),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 3, kernel_size=3, padding=1)
        )

    def forward(self, z):
        x = self.convTrans(z)
        return x


class ZUDecoder(nn.Module):
    def __init__(self, num_classes=2):
        super(ZUDecoder, self).__init__()
        self.fcTrans = nn.Sequential(
            nn.Linear(num_classes, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 512 * 30)
        )

        self.convTrans_1 = nn.Sequential(
            nn.Upsample(size=(13, 11), mode='bilinear'),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),
            nn.ConvTranspose2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),
            nn.ConvTranspose2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),
            nn.ConvTranspose2d(512, 512, kernel_size=3, padding=1),

            nn.Upsample(size=(27, 22), mode='bilinear'),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),
            nn.ConvTranspose2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),
            nn.ConvTranspose2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),
            nn.ConvTranspose2d(512, 256, kernel_size=3, padding=1),

            nn.Upsample(size=(54, 44), mode='bilinear'),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.ConvTranspose2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.ConvTranspose2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.ConvTranspose2d(256, 128, kernel_size=3, padding=1),
        )

        self.convTrans_2 = nn.Sequential(
            nn.Upsample(size=(109, 89), mode='bilinear'),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.ConvTranspose2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128, 64, kernel_size=3, padding=1),

            nn.Upsample(size=(218, 178), mode='bilinear'),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 3, kernel_size=3, padding=1)
        )

    def forward(self, z, u):
        u = self.fcTrans(u)
        u = u.view((u.size(0), 512, 6, 5))
        u = self.convTrans_1(u)

        z_u = torch.cat((z, u), dim=1)
        x = self.convTrans_2(z_u)

        return x
