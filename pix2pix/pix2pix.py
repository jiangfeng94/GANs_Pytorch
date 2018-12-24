import argparse
import os
from PIL import Image
import numpy as np
import torch,torchvision
import itertools
from torch.autograd import Variable
from torchvision.utils import save_image
from utils import weights_init,ImageDataset
from model import Generator,Discriminator
from torch.utils.data import DataLoader

parser =argparse.ArgumentParser()
parser.add_argument('--startepoch',type=int,default=0)
parser.add_argument('--batchsize', type=int, default=1, help='size of the batches')
parser.add_argument('--lr', type=float, default=0.0002, help='adam: learning rate')
parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
parser.add_argument('--img_height', type=int, default=256, help='size of image height')
parser.add_argument('--img_width', type=int, default=256, help='size of image width')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--sample_interval', type=int, default=50, help='interval between sampling images from generators')

opt = parser.parse_args()


gan_loss =torch.nn.MSELoss()
pixel_loss =torch.nn.L1Loss()
lambda_pixel = 100
Net_G =Generator()
Net_D =Discriminator()
Net_G.cuda()
Net_D.cuda()
gan_loss.cuda()
pixel_loss.cuda()
Net_G.apply(weights_init)
Net_D.apply(weights_init)
optimizer_G = torch.optim.Adam(Net_G.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(Net_D.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))


transforms_ = [ torchvision.transforms.Resize(int(opt.img_height*1.12), Image.BICUBIC),
                torchvision.transforms.RandomCrop((opt.img_height, opt.img_width)),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ]

# Training data loader
dataloader = DataLoader(ImageDataset("D:/project/赵老师的IDEA/CycleGAN/input/apple2orange", transforms_=transforms_, unaligned=True),
                        batch_size=opt.batchsize, shuffle=True)#, num_workers=opt.n_cpu)


FloatTensor = torch.cuda.FloatTensor
patch = (1, opt.img_height//2**4, opt.img_width//2**4)
for epoch in range(opt.startepoch, 100):
    for i, batch in enumerate(dataloader):
        real_A = Variable(batch['B'].type(FloatTensor))
        real_B = Variable(batch['A'].type(FloatTensor))
        valid = Variable(FloatTensor(np.ones((opt.batchsize, *patch))), requires_grad=False)
        fake = Variable(FloatTensor(np.zeros((opt.batchsize, *patch))), requires_grad=False)
        optimizer_G.zero_grad()

        # GAN loss
        fake_B = Net_G(real_A)
        loss_gan = gan_loss(Net_D(fake_B, real_A), valid)
        # Pixel-wise loss
        loss_pixel = pixel_loss(fake_B, real_B)
        g_loss =loss_gan +lambda_pixel * loss_pixel
        g_loss.backward()
        optimizer_G.step()

        optimizer_D.zero_grad()
        d_loss_real =gan_loss(Net_D(real_B,real_A),valid)
        d_loss_fake =gan_loss(Net_D(fake_B.detach(), real_A),fake)
        d_loss =0.5 *(d_loss_real+d_loss_fake)
        d_loss.backward()
        optimizer_D.step()
        print("[Epoch %d] [Batch %d/%d] [D loss: %f] [G loss: %f, adv: %f, pix: %f]" 
        %(epoch, i, len(dataloader), d_loss.item(), g_loss.item(),loss_gan.item(), loss_pixel.item()))
        
        batches_done = epoch * len(dataloader) + i
        if batches_done % opt.sample_interval == 0:
            img_sample = torch.cat((real_A.data, fake_B.data,real_B.data), 0)
            save_image(img_sample, '../../output/pix2pix_images/%s.png' % (batches_done), nrow=3, normalize=True)

