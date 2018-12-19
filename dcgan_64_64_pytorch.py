import argparse
import os
import glob
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from torchvision.utils import save_image
from torchvision import  transforms
parser = argparse.ArgumentParser()
parser.add_argument('--z_num',type=int,default=100,help='z_num')
parser.add_argument('--netD',default='',)
parser.add_argument('--netG',default='',)
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--batchsize', type=int, default=64, help='input batch size')
parser.add_argument('--img_size', type=int, default=64, help='imgsize')
opt = parser.parse_args()



def default_loader(path):
    img =Image.open(path).convert('RGB')
    img = img.resize((opt.img_size,opt.img_size))
    return img
class MyDataset(Dataset):
    def __init__(self, rootpath, transform=None, target_transform=None,loader=default_loader):
        imgs=glob.glob(rootpath)
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        fn = self.imgs[index]
        img = self.loader(fn)
        if self.transform is not None:
            img = self.transform(img)
        return img
    def __len__(self):
        return len(self.imgs)
train_data =MyDataset("D:/Desktop/data_1018/6/*.jpg",
            transform=transforms.Compose([
                               transforms.Resize(opt.img_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
dataloader = DataLoader(dataset=train_data, batch_size=64, shuffle=True)



def weights_init(m):
    classname =m.__class__.__name__
    if classname.find('Conv')!= -1:
        m.weight.data.normal_(0.0,0.02)
    elif classname.find('BatchNorm')!= -1:
        m.weight.data.normal_(1.0,0.02)
        m.bias.data.fill_(0)
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator,self).__init__()
        self.dis =nn.Sequential(
            # 3×64×64
            nn.Conv2d(3,64,4,2,1,bias=False),
            nn.LeakyReLU(0.2,inplace=True),
            # 64×32×32
            nn.Conv2d(64,128,4,2,1,bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2,inplace=True),
            # 128×16×16
            nn.Conv2d(128,256,4,2,1,bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2,inplace=True),
            # 256×8×8
            nn.Conv2d(256,512,4,2,1,bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2,inplace=True),
            # 512×4×4
            nn.Conv2d(512,1,4,1,0,bias=False),
            nn.Sigmoid()
        )
    def forward(self,input):
        output =self.dis(input)
        return output.view(-1,1).squeeze(1)

class Generator(nn.Module):
    def __init__(self):
        super(Generator,self).__init__()
        self.gen =nn.Sequential(
            nn.ConvTranspose2d(opt.z_num,512,4,1,0,bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # 512*4*4
            nn.ConvTranspose2d(512,256,4,2,1,bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # 256*8*8
            nn.ConvTranspose2d(256,128,4,2,1,bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # 128*16*16
            nn.ConvTranspose2d(128,64,4,2,1,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True), 
            # 64*32*32
            nn.ConvTranspose2d(64,3,4,2,1,bias=False),
            nn.Tanh()
            # 3*64*64
        )
    def forward(self,input):
        output =self.gen(input)
        return output

Net_G =Generator()
Net_D =Discriminator()

Net_G.apply(weights_init)
Net_D.apply(weights_init)
if opt.netD != '':
    Net_D.load_state_dict(torch.load(opt.netD))
if opt.netG !='':
    Net_G.load_state_dict(torch.load(opt.netG))

criterion = nn.BCELoss()
optimizerD = torch.optim.Adam(Net_D.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerG = torch.optim.Adam(Net_G.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
for epoch in range(200):
    for i,imgs in enumerate(dataloader):
        real_label = torch.full((opt.batchsize,), 1)
        fake_label = torch.full((opt.batchsize,), 0)
        # train errD_real
        Net_D.zero_grad()
        errD_real = criterion(Net_D(imgs), real_label)
        errD_real.backward()
        # train errD_fake
        z = torch.randn(opt.batchsize, opt.z_num, 1, 1)
        fake_imgs =Net_G(z)
        errD_fake = criterion(Net_D(fake_imgs.detach()), fake_label)
        # 这里的detach()函数非常的重要 
        errD_fake.backward()
        errD = errD_real + errD_fake
        optimizerD.step()
        #train errG
        Net_G.zero_grad()
        errG = criterion(Net_D(fake_imgs), real_label)
        errG.backward()
        optimizerG.step()
        print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f'
              % (epoch, 200, i, len(dataloader),errD.item(), errG.item()))
        if i % 10 == 0:
             save_image(fake_imgs.data[:25], '../output_64/%d_gen.png' % (epoch * len(dataloader) + i), nrow=5, normalize=True)
             save_image(imgs.data[:25], '../output_64/%d_real.png' % (epoch * len(dataloader) + i), nrow=5, normalize=True)