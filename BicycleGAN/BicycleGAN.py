import argparse
import torch,torchvision
import numpy as np
from torch.autograd import Variable
from torchvision.utils import save_image
from utils import weights_init,ImageDataset
from models import Generator,Discriminator,Encoder
from torch.utils.data import DataLoader
from PIL import Image
parse =argparse.ArgumentParser()
parse.add_argument("--epochs",type=int,default=100)
parse.add_argument("--batchsize",type=int,default=32)
parse.add_argument("--z_dim",type=int,default=100)
parse.add_argument("--lr",type=float,default=0.0002)
parse.add_argument("--b1",type=float,default=0.5)
parse.add_argument("--b2",type=float,default=0.999)
parse.add_argument('--img_height', type=int, default=128, help='size of image height')
parse.add_argument('--img_width', type=int, default=128, help='size of image width')
opt =parse.parse_args()
#init
Net_G =Generator(opt.z_dim,(3,opt.img_height,opt.img_width))
Net_E =Encoder(opt.z_dim)
D_VAE =Discriminator() 
D_LR  =Discriminator() 
l1_loss =torch.nn.L1Loss()
#cuda
Net_G.cuda()
Net_E.cuda()
D_VAE.cuda()
D_LR.cuda()
l1_loss.cuda()

#weight_init
Net_G.apply(weights_init)
D_VAE.apply(weights_init)
D_LR.apply(weights_init)
#optimizer
optimizer_E = torch.optim.Adam(Net_E.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_G = torch.optim.Adam(Net_G.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D_VAE = torch.optim.Adam(D_VAE.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D_LR = torch.optim.Adam(D_LR.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

FloatTensor =torch.cuda.FloatTensor

transforms_ = [ torchvision.transforms.Resize((opt.img_height, opt.img_width), Image.BICUBIC),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ]
dataloader = DataLoader(ImageDataset("D:/project/赵老师的IDEA/CycleGAN/input/apple2orange", transforms_=transforms_, unaligned=True),
                        batch_size=opt.batchsize, shuffle=True)#, num_workers=opt.n_cpu)

def reparameterization(mu, logvar):
    std = torch.exp(logvar / 2)
    sampled_z = Variable(FloatTensor(np.random.normal(0, 1, (mu.size(0), opt.z_dim))))
    z = sampled_z * std + mu
    return z
for epoch in range(opt.epochs):
    for i,batch in enumerate(dataloader):
        real_A =Variable(batch['A'].type(FloatTensor))
        real_B =Variable(batch['B'].type(FloatTensor))
        valid = 1
        fake = 0
        optimizer_E.zero_grad()
        optimizer_G.zero_grad()
        # cVAE-GAN
        mu, logvar = Net_E(real_B)
        encoded_z = reparameterization(mu, logvar)
        fake_B =Net_G(real_A,encoded_z)
        loss_pixel =l1_loss(fake_B,real_B)
        loss_kl = torch.sum(0.5 * (mu**2 + torch.exp(logvar) - logvar - 1))
        loss_VAE_GAN = D_VAE.compute_loss(fake_B, valid)
        #cLR-GAN
        sample_z =Variable(FloatTensor(np.random.normal(0,1,(opt.batchsize,opt.z_dim))))
        fake_B_ =Net_G(real_A,sample_z)
        loss_LR_GAN = D_LR.compute_loss(fake_B_,valid)
        #G_E loss
        loss_GE =loss_VAE_GAN +loss_LR_GAN+ 10 *loss_pixel + 0.01 *loss_kl
        loss_GE.backward(retain_graph=True)
        optimizer_E.step()
        
        _mu, _ = Net_E(fake_B_)
        loss_latent = 0.5 * l1_loss(_mu, sample_z)
        loss_latent.backward()
        optimizer_G.step()

        #D_VAE
        optimizer_D_VAE.zero_grad()
        loss_D_VAE = D_VAE.compute_loss(real_B, valid) +  D_VAE.compute_loss(fake_B.detach(), fake)
        loss_D_VAE.backward()
        optimizer_D_VAE.step()
        #D_LR
        optimizer_D_LR.zero_grad()
        loss_D_LR = D_LR.compute_loss(real_B, valid) + D_LR.compute_loss(fake_B_.detach(), fake)
        loss_D_LR.backward()
        optimizer_D_LR.step()

        batches_done = epoch * len(dataloader) + i
        print("[Epoch %d] [Batch %d/%d] [D_VAE loss: %f D_LR loss: %f] [G loss: %f, pix: %f, latent: %f]" % (epoch, i, len(dataloader), loss_D_VAE.item(), loss_D_LR.item(),loss_GE.item(), loss_pixel.item(),loss_latent.item()))

        if batches_done % 50 == 0:
            img_sample = torch.cat((real_A.data, fake_B.data,real_B.data), 0)
            save_image(img_sample, '../../output/bicyclegan/%s.png' % (batches_done), nrow=3, normalize=True)

