# Script to train the network

from model import generator
from torch.autograd import Variable
import torch


def train():
    """
    Function to train the network
    """
    raise NotImplementedError


def evaluate(net, z):
    """
    Function to evaluate the generator network.
    """
    net.eval()
    z = z.resize_(1, 100, 1, 1)
    z = Variable(z)
    output = net(z)
    return output


if __name__ == '__main__':
    net = generator(input_size=100, output_size=64)
    z = torch.rand(1, 100)
    output = evaluate(net, z)
    img = output.data.cpu().numpy()
    print img.shape
