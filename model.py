# Contains the definition of the generator and discriminator networks.

import torch

import torch.nn as nn
import torch.nn.functional as F


class generator(nn.Module):
    def __init__(self, input_size, output_size):
        super(generator, self).__init__()
        self.conv_tp1 = nn.ConvTranspose2d(100, 512, kernel_size=4, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(512)
        # TODO : Define this in a loop, don't hardcode layer sizes
        self.conv_tp2 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(256)
        self.conv_tp3 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv_tp4 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(64)
        self.conv_tp5 = nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1, bias=False)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv_tp1(x)))
        out = F.relu(self.bn2(self.conv_tp2(out)))
        out = F.relu(self.bn3(self.conv_tp3(out)))
        out = F.relu(self.bn4(self.conv_tp4(out)))
        out = F.relu(self.conv_tp5(out))
        return torch.tanh(out)


# class DCGAN(nn.Module):
#     def __init__(self, gen_params, disc_params):
#         super(DCGAN, self).__init__()
