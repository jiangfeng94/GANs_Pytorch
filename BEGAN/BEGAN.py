import argparse
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision.utils import save_image
import os
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
parser =argparse.ArgumentParser()
parser.add_argument('--totalepochs',type=int,default=100)
parser.add_argument('--batchsize',type=int,default=64)
parser.add_argument('--lr',type=float,default=0.0002)
parser.add_argument('--b1',type=float,default=0.5)
parser.add_argument('--b2',type=float,default=0.999)
parser.add_argument('--z_dim',type=int,default=100)
parser.add_argument('--channels',type=int,default=3)
parser.add_argument('--img_size',type=int,default=32)
opt =parser.parse_args()
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)
img_shape =(opt.channels,opt.img_size,opt.img_size)

class Generator(nn.Module):
    def __init__(self):
        super(Generator,self).__init__()
        self.init_size =opt.img_size//4
        self.l1 = nn.Sequential(nn.Linear(opt.z_dim,128*self.init_size**2))
        self.gen =nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, opt.channels, 3, stride=1, padding=1),
            nn.Tanh()
        )
    def forward(self,z):
        out =self.l1(z)
        out =out.view(opt.batchsize,128,self.init_size,self.init_size)
        return self.gen(out)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator,self).__init__()
        self.down =nn.Sequential(
            nn.Conv2d(opt.channels,64,3,2,1),
            nn.ReLU()
        )
        self.down_size =opt.img_size//2
        down_dim =64*(opt.img_size//2)**2
        self.fc =nn.Sequential(
            nn.Linear(down_dim,32),
            nn.BatchNorm1d(32,0.8),
            nn.ReLU(inplace=True),
            nn.Linear(32,down_dim),
            nn.BatchNorm1d(down_dim),
            nn.ReLU(inplace=True)
        )
        self.up =nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64,opt.channels,3,1,1)
        )
    def forward(self,img):
        out =self.down(img)
        out =self.fc(out.view(out.size(0), -1))
        out =self.up(out.view(out.size(0),64,self.down_size,self.down_size))
        return out

G =Generator()
D =Discriminator()
G.cuda()
D.cuda()
G.apply(weights_init)
D.apply(weights_init)
optimizer_G = torch.optim.Adam(G.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(D.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
FloatTensor=torch.cuda.FloatTensor

transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize((opt.img_size,opt.img_size)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), ])
dataset = torchvision.datasets.ImageFolder("D:/DATASET/celeba", transform=transforms)
dataloader = torch.utils.data.DataLoader(
    dataset=dataset,
    batch_size=opt.batchsize,
    shuffle=True,
    drop_last=True,
)


gamma = 0.75
lambda_k = 0.001
k = 0.

for epoch in range(opt.totalepochs):
    for i,(imgs,_) in enumerate(dataloader):
        real_imgs =Variable(imgs.type(FloatTensor))
        z = Variable(FloatTensor(np.random.normal(0, 1, (opt.batchsize, opt.z_dim))))
        optimizer_G.zero_grad()
        gen_imgs =G(z)
        g_loss =torch.mean(torch.abs(D(gen_imgs)-gen_imgs))
        g_loss.backward()
        optimizer_G.step()

        optimizer_D.zero_grad()
        d_real =D(real_imgs)
        d_fake =D(gen_imgs.detach())
        d_loss_real =torch.mean(torch.abs(d_real-real_imgs))
        d_loss_fake =torch.mean(torch.abs(d_fake-gen_imgs.detach()))
        
        d_loss =d_loss_real -k*d_loss_fake
        d_loss.backward()
        optimizer_D.step()

        diff =torch.mean(gamma*d_loss_real -d_loss_fake)
        k =k +lambda_k*diff.item()
        k =min(max(k,0),1) 
        M =(d_loss_real +torch.abs(diff)).data
        print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] -- M: %f, k: %f" % (epoch, opt.totalepochs, i, len(dataloader),
                                                            d_loss.item(), g_loss.item(),M, k))
        batches_done = epoch * len(dataloader) + i
        if batches_done % 10 == 0:
            os.makedirs('../../output/began_output',exist_ok=True)
            save_image(gen_imgs.data[:25], '../../output/began_output/%d.png' % batches_done, nrow=5, normalize=True)

