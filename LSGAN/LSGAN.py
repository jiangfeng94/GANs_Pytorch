
import argparse
import os
import numpy as np
import torch,torchvision
import torch.nn as nn
from torchvision.utils import save_image
from torch.autograd import Variable
parser =argparse.ArgumentParser()
parser.add_argument('--data_path',default ='D:/Desktop/celeba')
parser.add_argument('--img_size',type=int,default=32)
parser.add_argument('--batchsize',type=int,default=128)
parser.add_argument('--netD',default='',)
parser.add_argument('--netG',default='',)
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--z_num',type=int,default=100)
opt =parser.parse_args()

transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize((opt.img_size,opt.img_size)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), ])
dataset = torchvision.datasets.ImageFolder(opt.data_path, transform=transforms)
dataloader = torch.utils.data.DataLoader(
    dataset=dataset,
    batch_size=opt.batchsize,
    shuffle=True,
    drop_last=True,
)
def weights_init(m):
    classname =m.__class__.__name__
    if classname.find('Conv')!= -1:
        m.weight.data.normal_(0.0,0.02)
    elif classname.find('BatchNorm')!= -1:
        m.weight.data.normal_(1.0,0.02)
        m.bias.data.fill_(0)
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.init_size = opt.img_size // 4
        self.l1 = nn.Sequential(nn.Linear(opt.z_num, 128*self.init_size**2))

        self.conv_blocks = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 3, 3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [   nn.Conv2d(in_filters, out_filters, 3, 2, 1),
                        nn.LeakyReLU(0.2, inplace=True),
                        nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(3, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = opt.img_size // 2**4
        self.adv_layer = nn.Linear(128*ds_size**2, 1)

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)

        return validity
# 关键处 MSELOSS 替代 BECLOSS
adversarial_loss = torch.nn.MSELoss()
#adversarial_loss = nn.BCELoss()

Net_G =Generator()
Net_D =Discriminator()
Net_D.cuda()
Net_G.cuda()
adversarial_loss.cuda()
Net_G.apply(weights_init)
Net_D.apply(weights_init)
#模型的读取
if opt.netD != '':
    Net_D.load_state_dict(torch.load(opt.netD))
if opt.netG !='':
    Net_G.load_state_dict(torch.load(opt.netG))


optimizerD = torch.optim.Adam(Net_D.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerG = torch.optim.Adam(Net_G.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
FloatTensor =torch.cuda.FloatTensor

for epoch in range(200):
    for i,(imgs,label) in enumerate(dataloader):
        valid = Variable(FloatTensor(imgs.shape[0], 1).fill_(1.0), requires_grad=False)
        fake = Variable(FloatTensor(imgs.shape[0], 1).fill_(0.0), requires_grad=False)
        real_imgs =Variable(imgs.type(FloatTensor))
        # train errD_real
        Net_D.zero_grad()
        errD_real = adversarial_loss(Net_D(real_imgs), valid)
        errD_real.backward()
        # train errD_fake
        z = Variable(FloatTensor(np.random.normal(0, 1, (opt.batchsize, opt.z_num))))
        fake_imgs =Net_G(z)
        errD_fake = adversarial_loss(Net_D(fake_imgs.detach()), fake)
        # 这里的detach()函数非常的重要 
        errD_fake.backward()
        errD = errD_real + errD_fake
        optimizerD.step()
        #train errG
        Net_G.zero_grad()
        errG = adversarial_loss(Net_D(fake_imgs), valid)
        errG.backward()
        optimizerG.step()
        print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f'
              % (epoch, 200, i, len(dataloader),errD.item(), errG.item()))
        if i % 10 == 0:
             if not os.path.exists('../../output/LSGAN_CelebA'):
                os.mkdir('../../output/LSGAN_CelebA')
             save_image(fake_imgs.data[:25], '../../output/LSGAN_CelebA/%d_gen.png' % (epoch * len(dataloader) + i), nrow=5, normalize=True)
             save_image(imgs.data[:25], '../../output/LSGAN_CelebA/%d_real.png' % (epoch * len(dataloader) + i), nrow=5, normalize=True)
    # 模型的保存
    torch.save(Net_G.state_dict(), '../../output/LSGAN_CelebA/netG_epoch_%d.pth' % (epoch))
    torch.save(Net_D.state_dict(), '../../output/LSGAN_CelebA/netD_epoch_%d.pth' % (epoch))

