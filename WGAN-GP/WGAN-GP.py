import argparse
import os
import numpy as np
import torch,torchvision
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms as transforms
from torchvision.utils import save_image
import torch.nn as nn
from torch.autograd import Variable
parser = argparse.ArgumentParser()
parser.add_argument('--batchsize', type=int, default=64, help='size of the batches')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate')
parser.add_argument('--z_dim', type=int, default=100, help='dimensionality of the latent space')
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
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

lambda_gp =10
G =Generator()
D =Discriminator()
G.cuda()
D.cuda()
optimizer_G = torch.optim.Adam(G.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(D.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
FloatTensor =torch.cuda.FloatTensor

def compute_gradient_penalty(D, real_img, fake_img):
    alpha = FloatTensor(np.random.random((opt.batchsize,1,1,1)))
    interpolates =(alpha*real_img +(1-alpha)*fake_img).requires_grad_(True)
    d_interpolates=D(interpolates)
    fake =Variable(FloatTensor(opt.batchsize,1).fill_(1.0),requires_grad=False)
    gradients =torch.autograd.grad(
        outputs =d_interpolates,
        inputs =interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


batches_done = 0
for epoch in range(100):
    for i,(imgs,_) in enumerate(dataloader):
        real_imgs=Variable(imgs.type(FloatTensor))
        optimizer_D.zero_grad()
        z = Variable(FloatTensor(np.random.normal(0, 1, (opt.batchsize, opt.z_dim))))
        fake_imgs = G(z)
        #gp
        gradient_penalty =compute_gradient_penalty(D, real_imgs.data, fake_imgs.data)
        loss_D = -torch.mean(D(real_imgs)) + torch.mean(D(fake_imgs))+lambda_gp*gradient_penalty
        loss_D.backward()
        optimizer_D.step()
        optimizer_G.zero_grad()
        if i % opt.n_critic == 0:
            
            gen_imgs =G(z)
            loss_G =-torch.mean(D(gen_imgs))
            loss_G.backward()
            optimizer_G.step()

            print ("[Epoch %d] [Batch %d/%d] [D loss: %f] [G loss: %f]" % (epoch, 
                                                            batches_done % len(dataloader), len(dataloader),
                                                            loss_D.item(), loss_G.item()))
        if batches_done % 100 == 0:
            if not os.path.exists('../../output/mnist_wgan-pg'):
                os.mkdir('../../output/mnist_wgan-pg')
            save_image(gen_imgs.data[:64], '../../output/mnist_wgan-pg/%d.png' % (epoch * len(dataloader) + i), nrow=8, normalize=True)
        batches_done += 1