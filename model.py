# Contains the definition of the generator and discriminator networks.

import torch

import torch.nn as nn
import torch.nn.functional as F


class generator(nn.Module):
    def __init__(self, input_size, output_size):
        super(generator, self).__init__()
        self.linear = nn.Linear(input_size, 4*4*1024)
        self.conv1 = nn.ConvTranspose2d()

    def forward(self, x):
        out = self.linear(x)
class DCGAN(nn.Module):
    def __init__(self, gen_params, disc_params):
        super(DCGAN, self).__init__()
