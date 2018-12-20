import argparse
import numpy as np
import torch,torchvision
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms as transforms
from torchvision.utils import save_image
import torch.nn as nn
from torch.autograd import Variable
parser = argparse.ArgumentParser()
parser.add_argument('--batchsize', type=int, default=256, help='size of the batches')
parser.add_argument('--lr', type=float, default=0.00005, help='learning rate')
parser.add_argument('--z_dim', type=int, default=100, help='dimensionality of the latent space')
parser.add_argument('--img_size', type=int, default=28, help='size of each image dimension')
parser.add_argument('--channels', type=int, default=1, help='number of image channels')
parser.add_argument('--n_critic', type=int, default=5, help='number of training steps for discriminator per iter')
parser.add_argument('--clip_value', type=float, default=0.01, help='lower and upper clip value for disc. weights')
opt = parser.parse_args()

dataset = datasets.MNIST('../../mnist_data', train=True, download=True,
                   transform=transforms.Compose([
                        transforms.Resize(opt.img_size),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                   ]))
dataloader = DataLoader(dataset,batch_size=opt.batchsize, drop_last=True,shuffle=True)

img_shape = (opt.channels, opt.img_size, opt.img_size)
class Generator(nn.Module):
    def __init__(self):
        super(Generator,self).__init__()
        def block(in_feat,out_feat,normalize=True):
            layers =[nn.Linear(in_feat,out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat,0.8))
            layers.append(nn.LeakyReLU(0.2,inplace=True))
            return layers
        self.gen =nn.Sequential(
            *block(opt.z_dim,128,normalize=False),
            *block(128,256),
            *block(256,512),
            *block(512,1024),
            nn.Linear(1024,int(np.prod(img_shape))),
            nn.Tanh()
        )
    def forward(self,z):
        img =self.gen(z)
        img =img.view(img.shape[0],*img_shape)
        return img
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator,self).__init__()
        self.dis =nn.Sequential(
            nn.Linear(int(np.prod(img_shape)),512),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Linear(512,256),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Linear(256,1)
        )
        #原先GAN的判别器是一个二分类问题
        #而wgan的判别器是个回归问题，所以取消了最后一层的sigmoid层
    def forward(self,img):
        img =img.view(img.shape[0],-1)
        return self.dis(img)

Net_D =Discriminator()
Net_G =Generator()
Net_D.cuda()
Net_G.cuda()

# 优化器
# 论文中提出不适用带有动量的优化器(Adam)
# 使用RMSprop、SGD等
# 没有理论上的证明，玄学-。-
optimizer_G = torch.optim.RMSprop(Net_G.parameters(), lr=opt.lr)
optimizer_D = torch.optim.RMSprop(Net_D.parameters(), lr=opt.lr)
FloatTensor = torch.cuda.FloatTensor
batches_done = 0
for epoch in range(100):
    for i,(imgs,_) in enumerate(dataloader):
        real_imgs=Variable(imgs.type(FloatTensor))
        optimizer_D.zero_grad()
        z = Variable(FloatTensor(np.random.normal(0, 1, (opt.batchsize, opt.z_dim))))
        fake_imgs = Net_G(z).detach()
        loss_D = -torch.mean(Net_D(real_imgs)) + torch.mean(Net_D(fake_imgs))
        loss_D.backward()
        optimizer_D.step()
        #clip weight 
        #每次更新判别器的参数之后把它们的绝对值截断到不超过一个固定常数c
        # 每当更新完一次判别器的参数之后，就检查判别器的所有参数的值有没有超过一个阈值，比如0.01，
        # 有的话就把这些参数clip回 [-0.01, 0.01] 范围里
        # 通过在训练过程中保证判别器的所有参数有界
        # 就保证了判别器不能对两个略微不同的样本给出天差地别的分数值，从而间接实现了Lipschitz限制。
        for p in Net_D.parameters():
            p.data.clamp_(-opt.clip_value, opt.clip_value)
        if i % opt.n_critic == 0:
            optimizer_G.zero_grad()
            gen_imgs =Net_G(z)
            loss_G =-torch.mean(Net_D(gen_imgs))
            loss_G.backward()
            optimizer_G.step()

            print ("[Epoch %d] [Batch %d/%d] [D loss: %f] [G loss: %f]" % (epoch, 
                                                            batches_done % len(dataloader), len(dataloader),
                                                            loss_D.item(), loss_G.item()))
        if batches_done % 100 == 0:
            save_image(gen_imgs.data[:100], '../../mnist_wgan/%d.png' % (epoch * len(dataloader) + i), nrow=10, normalize=True)
        batches_done += 1
