# Script to train the network

from model import generator, discriminator
from torch.autograd import Variable
import torch
from matplotlib import pyplot as plt


def train():
    """
    Function to train the network
    """
    raise NotImplementedError


def g_evaluate(net, z):
    """
    Function to evaluate the generator network.
    """
    net.eval()
    z = z.resize_(1, 100, 1, 1)
    z = Variable(z)
    output = net(z)
    return output

def d_evaluate(net, img):
    """
    Evaluate the discriminator network
    """
    net.eval()
    # img = Variable(img)
    output = net(img)
    return output

if __name__ == '__main__':
    g_net = generator(input_size=100, output_size=64)
    batch_size = 1
    z = torch.rand(batch_size, 100)
    g_output = g_evaluate(g_net, z)
    print g_output.size()

    d_net = discriminator(image_size=64)
    d_output = d_evaluate(d_net, g_output)
    print d_output.data.cpu().numpy()
    # output = output.data.cpu().numpy()
    # img = output[0].T
    # print img*255
    # plt.imshow(img*255.0)
    # plt.show()
