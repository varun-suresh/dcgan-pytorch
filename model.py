# Contains the definition of the generator and discriminator networks.

import torch

import torch.nn as nn
import torch.nn.functional as F


class generator(nn.Module):
    def __init__(self, input_size, output_size):
        super(generator, self).__init__()
        self.conv_tp1 = nn.ConvTranspose2d(100, 512, kernel_size=4, stride=1, bias=False)
        self.bn_g1 = nn.BatchNorm2d(512)
        # TODO : Define this in a loop, don't hardcode layer sizes
        self.conv_tp2 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn_g2 = nn.BatchNorm2d(256)
        self.conv_tp3 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn_g3 = nn.BatchNorm2d(128)
        self.conv_tp4 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn_g4 = nn.BatchNorm2d(64)
        self.conv_tp5 = nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1, bias=False)

    def forward(self, x):
        out = F.relu(self.bn_g1(self.conv_tp1(x)))
        out = F.relu(self.bn_g2(self.conv_tp2(out)))
        out = F.relu(self.bn_g3(self.conv_tp3(out)))
        out = F.relu(self.bn_g4(self.conv_tp4(out)))
        out = F.relu(self.conv_tp5(out))
        return torch.tanh(out)


class discriminator(nn.Module):
    def __init__(self, image_size):
        super(discriminator, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1, bias=False)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn_d2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn_d3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn_d4 = nn.BatchNorm2d(512)
        self.conv5 = nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0, bias=False)

    def forward(self, img):
        out = F.leaky_relu(self.conv1(img), negative_slope=0.02, inplace=True)
        out = F.leaky_relu(self.bn_d2(self.conv2(out)), negative_slope=0.02, inplace=True)
        out = F.leaky_relu(self.bn_d3(self.conv3(out)), negative_slope=0.02, inplace=True)
        out = F.leaky_relu(self.bn_d4(self.conv4(out)), negative_slope=0.02, inplace=True)
        out = F.sigmoid(self.conv5(out))
        return out

# class DCGAN(nn.Module):
#     def __init__(self, gen_params, disc_params):
#         super(DCGAN, self).__init__()
