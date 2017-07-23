# Contains the definition of the generator and discriminator networks.

import torch

import torch.nn as nn
import torch.nn.functional as F


class generator(nn.Module):
    def __init__(self, input_size, output_size):
        super(generator, self).__init__()
        # TODO : Define this in a loop, don't hardcode layer sizes
        self.conv1 = nn.ConvTranspose2d(100, 512, kernel_size=4, stride=1, bias=False)
        self.conv2 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False)
        self.conv3 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False)
        self.conv4 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False)
        self.conv5 = nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1, bias=False)

    def forward(self, x):
        # out = self.linear(x)
        # out = out.view((1, -1, 4, 4))
        out = self.conv1(x)
        out = F.relu(out)
        print out.size()
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv3(out))
        out = F.relu(self.conv4(out))
        out = F.relu(self.conv5(out))
        return torch.tanh(out)


# class DCGAN(nn.Module):
#     def __init__(self, gen_params, disc_params):
#         super(DCGAN, self).__init__()
