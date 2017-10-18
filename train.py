# Script to train the network

from model import generator, discriminator
from torch.autograd import Variable
import torch
from torch import nn
from torch import optim
from torchvision import utils as vutils
from torchvision import transforms
from torchvision import datasets as dset
from matplotlib import pyplot as plt
import argparse

def train(dataloader, out_dir, n_iter, batch_size, img_size, nz, lr, beta1, d_net, g_net):
    """
    Function to train the network
    """
    criterion = nn.BCELoss()
    real_label = 1
    fake_label = 0
    input_vec = torch.FloatTensor(batch_size, 3, img_size, img_size)
    noise = torch.FloatTensor(batch_size, nz, 1, 1)
    # Use this to generate fake data for visualization
    fixed_noise = torch.FloatTensor(batch_size, nz, 1, 1)
    fixed_noise = Variable(fixed_noise)
    label = torch.FloatTensor(batch_size)

    optimizer_d = optim.Adam(d_net.parameters(), lr, betas=(beta1, 0.999))
    optimizer_g = optim.Adam(g_net.parameters(), lr, betas=(beta1, 0.999))
    for epoch in range(n_iter):
        for i, data in enumerate(dataloader, 0):
            # Update D network :
            # Real data:
            d_net.zero_grad()
            real_cpu, _ = data
            batch_size = real_cpu.size(0)
            input_vec.resize_as_(real_cpu).copy_(real_cpu)
            label.resize_(batch_size).fill_(real_label)
            input_var = Variable(input_vec)
            label_var = Variable(label)
            output = d_net(input_var)
            err_d_real = criterion(output, label_var)
            err_d_real.backward()
            D_x = output.data.mean()
            # Fake data:
            noise.resize_(batch_size, nz, 1, 1).normal_(1, 1)
            noise_var = Variable(noise)
            fake_var = g_net(noise_var)
            label_var = Variable(label.fill_(fake_label))
            output = d_net(fake_var.detach())
            error_d_fake = criterion(output, label_var)
            error_d_fake.backward()
            D_G_z1 = output.data.mean()
            err_d = err_d_real + error_d_fake
            optimizer_d.step()
            # Update G network:
            g_net.zero_grad()
            label_var = Variable(label.fill_(real_label))
            output = d_net(fake_var)
            err_g = criterion(output, label_var)
            D_G_z2 = output.data.mean()
            err_g.backward()
            optimizer_g.step()

            print '[%d/%d] [%d/%d] Loss_D: %.6f Loss_G: %.6f D(x): %.6f D(G(z)) : %.6f / %.6f' %(epoch, n_iter, i, len(dataloader), err_d.data[0],err_g.data[0],D_x, D_G_z1, D_G_z2)

            if i % 100 == 0 :
                vutils.save_image(real_cpu, '%s/real_samples.png'%out_dir, normalize=True)
                fake = g_net(fixed_noise)
                vutils.save_image(fake.data, '%s/fake_samples_%03d.png'%(out_dir, epoch), normalize=True)

            # Save the checkpoint
            torch.save(d_net.state_dict(), '%s/net_d_epoch_%03d.pth'%(out_dir, epoch))
            torch.save(g_net.state_dict(), '%s/net_g_epoch_%03d.pth'%(out_dir, epoch))

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
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_iter',
                        type=int,
                        help='Number of iterations',
                        default=10)
    parser.add_argument('--batch_size',
                        type=int,
                        help='Number of images in a batch',
                        default=100)
    parser.add_argument('--lr',
                        type=float,
                        help='Learning rate',
                        default=0.0002)
    parser.add_argument('--nz',
                        type=int,
                        help='Size of the latent variable',
                        default=100)
    parser.add_argument('--out_dir',
                        type=str,
                        help='Directory to store checkpoints',
                        default='/home/varun/pytorch-projects/dcgan-pytorch')
    parser.add_argument('--img_size',
                        type=int,
                        help='Size of the input image',
                        default=64)
    parser.add_argument('--beta1',
                        type=float,
                        help='Adam optimizer parameter',
                        default=0.5)
    args = parser.parse_args()
    dataset = dset.MNIST(root='/home/varun', transform=transforms.Compose([
                                   transforms.Scale(args.img_size),
                                   transforms.CenterCrop(args.img_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                             shuffle=True)
    g_net = generator(input_size=100, output_size=64)
    d_net = discriminator(image_size=64)

    train(dataloader,
          args.out_dir,
          args.n_iter,
          args.batch_size,
          args.img_size, \
          args.nz,
          args.lr,
          args.beta1,
          d_net,
          g_net)

    # z = torch.rand(args.batch_size, 100)
    # g_output = g_evaluate(g_net, z)
    # print g_output.size()
    #
    #
    # d_output = d_evaluate(d_net, g_output)
    # print d_output.data.cpu().numpy()
    # output = g_output.data.cpu().numpy()
    # img = output[0].T
    # print img*255
    # plt.imshow(img*255.0)
    # plt.show()
