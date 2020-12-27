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
