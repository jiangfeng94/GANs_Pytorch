import argparse
import os
import numpy as np
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torch.nn as nn
import torch
from models import Generator,Discriminator

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

parser = argparse.ArgumentParser()
parser.add_argument('--total_epochs', type=int, default=200, help='number of epochs of training')
parser.add_argument('--batchsize', type=int, default=32, help='size of the batches')
parser.add_argument('--lr', type=float, default=0.0002, help='adam: learning rate')
parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
parser.add_argument('--z_dim', type=int, default=100, help='dimensionality of the latent space')
parser.add_argument('--img_size', type=int, default=32, help='size of each image dimension')
parser.add_argument('--channels', type=int, default=3, help='number of image channels')
parser.add_argument('--sample_interval', type=int, default=400, help='interval betwen image samples')
opt = parser.parse_args()

img_shape = (opt.channels, opt.img_size, opt.img_size)
gan_loss = torch.nn.MSELoss()
Net_G = Generator()
Net_D = Discriminator()
gan_loss.cuda()
Net_D.cuda()
Net_G.cuda()
Net_D.apply(weights_init)
Net_G.apply(weights_init)

optimizer_G = torch.optim.Adam(Net_G.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(Net_D.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

FloatTensor = torch.cuda.FloatTensor 

for epoch in range(opt.n_epochs):
    for i, ((imgsA, _), (imgsB, _)) in enumerate(zip(dataloader1, dataloader2)):
        valid = Variable(FloatTensor(opt.batchsize, 1).fill_(1.0), requires_grad=False)
        fake = Variable(FloatTensor(opt.batch_size, 1).fill_(0.0), requires_grad=False)
        imgsA = Variable(imgsA.type(FloatTensor))#.expand(imgs1.size(0), 3, opt.img_size, opt.img_size))
        imgsB = Variable(imgsB.type(FloatTensor))

        optimizer_G.zero_grad()
        z = Variable(FloatTensor(np.random.normal(0, 1, (opt.batchsize, opt.z_dim))))
        gen_A, gen_B = Net_G(z)
        validity_A, validity_B = Net_D(gen_A, gen_B)
        g_loss = (gan_loss(validity_A, valid) + gan_loss(validity_B, valid)) / 2
        g_loss.backward()
        optimizer_G.step()

        optimizer_D.zero_grad()
        validity_A_real, validity_B_real = Net_D(imgsA, imgsB)
        validity_A_fake, validity_B_fake = Net_D(gen_A.detach(), gen_B.detach())

        d_loss =    (gan_loss(validity_A_real, valid) + \
                    gan_loss(validity_A_fake, fake) + \
                    gan_loss(validity_B_real, valid) + \
                    gan_loss(validity_B_fake, fake)) / 4
        d_loss.backward()
        optimizer_D.step()
        print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" % (epoch, opt.n_epochs, i, len(dataloader1),
                                                            d_loss.item(), g_loss.item()))
