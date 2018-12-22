import argparse
import os
from PIL import Image
import numpy as np
import torch,torchvision
import itertools
from torch.autograd import Variable
from utils import weights_init,ImageDataset,ReplayBuffer
from models import Generator,Discriminator
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
opt = parser.parse_args()
# Loss
gan_loss =torch.nn.MSELoss()
cycle_loss =torch.nn.L1Loss()
identity_loss =torch.nn.L1Loss()
# NetWork
G_A2B = Generator()
G_B2A = Generator()
D_A =Discriminator()
D_B =Discriminator()
# gpu cuda()
G_A2B.cuda()
G_B2A.cuda()
D_A.cuda()
D_B.cuda()
gan_loss.cuda()
cycle_loss.cuda()
identity_loss.cuda()

if opt.startepoch !=0:
    G_A2B.load_state_dict(torch.load('saved_models/%s/G_AB_%d.pth' % (opt.dataset_name, opt.startepoch)))
    G_B2A.load_state_dict(torch.load('saved_models/%s/G_BA_%d.pth' % (opt.dataset_name, opt.startepoch)))
    D_A.load_state_dict(torch.load('saved_models/%s/D_A_%d.pth' % (opt.dataset_name, opt.startepoch)))
    D_B.load_state_dict(torch.load('saved_models/%s/D_B_%d.pth' % (opt.dataset_name, opt.startepoch)))
else:
    G_A2B.apply(weights_init)
    G_B2A.apply(weights_init)
    D_A.apply(weights_init)
    D_B.apply(weights_init)

# loss weights
lambda_cycle =10
lambda_identity =0.5*lambda_cycle


#optimizers
optimizer_G = torch.optim.Adam(itertools.chain(G_A2B.parameters(), G_B2A.parameters()),lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D_A = torch.optim.Adam(D_A.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D_B = torch.optim.Adam(D_B.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
#缓存
fake_A_buffer = ReplayBuffer()
fake_B_buffer = ReplayBuffer()


transforms_ = [ torchvision.transforms.Resize(int(opt.img_height*1.12), Image.BICUBIC),
                torchvision.transforms.RandomCrop((opt.img_height, opt.img_width)),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ]

# Training data loader
dataloader = DataLoader(ImageDataset("D:/project/赵老师的IDEA/CycleGAN/input/apple2orange", transforms_=transforms_, unaligned=True),
                        batch_size=opt.batchsize, shuffle=True)#, num_workers=opt.n_cpu)


FloatTensor = torch.cuda.FloatTensor
#output of image discriminator
patch =(1, opt.img_height // 2**4, opt.img_width // 2**4)
for epoch in range(opt.startepoch,100):
    for i,batch in enumerate(dataloader):
        real_A =Variable(batch['A'].type(FloatTensor))
        real_B =Variable(batch['B'].type(FloatTensor))
        valid =Variable(FloatTensor(np.ones((opt.batchsize,*patch))), requires_grad=False)
        fake =Variable(FloatTensor(np.zeros((opt.batchsize,*patch))),requires_grad=False)
        #### Train G ####
        optimizer_G.zero_grad()
        #id_loss
        loss_id_A = identity_loss(G_B2A(real_A), real_A)
        loss_id_B = identity_loss(G_A2B(real_B), real_B)
        loss_id =(loss_id_A+loss_id_B)/2
        #gan_loss
        fake_B =G_A2B(real_A)
        loss_G_A2B =gan_loss(D_B(fake_B),valid)
        fake_A =G_B2A(real_B)
        loss_G_B2A =gan_loss(D_A(fake_A),valid)
        loss_GAN =(loss_G_A2B+loss_G_B2A)/2
        #cycle_loss
        re_G_A =G_B2A(fake_B)
        loss_cycle_A =cycle_loss(re_G_A,real_A)
        re_G_B =G_A2B(fake_A)
        loss_cycle_B =cycle_loss(re_G_B,real_B)
        loss_cycle =(loss_cycle_A+loss_cycle_B)/2
        #total_loss
        loss_G = loss_GAN + lambda_cycle * loss_cycle +lambda_identity * loss_id
        loss_G.backward()
        optimizer_G.step()

        #### Train D A ####
        optimizer_D_A.zero_grad()
        loss_real = gan_loss(D_A(real_A), valid)
        fake_A_ = fake_A_buffer.push_and_pop(fake_A)
        loss_fake = gan_loss(D_A(fake_A_.detach()), fake)
        loss_D_A = (loss_real + loss_fake) / 2
        loss_D_A.backward()
        optimizer_D_A.step()

        #### Train D B ####
        optimizer_D_B.zero_grad()
        loss_real = gan_loss(D_B(real_B), valid)
        fake_B_ = fake_B_buffer.push_and_pop(fake_B)
        loss_fake = gan_loss(D_B(fake_B_.detach()), fake)
        loss_D_B = (loss_real + loss_fake) / 2
        loss_D_B.backward()
        optimizer_D_B.step()
        loss_D = (loss_D_A + loss_D_B) / 2
        print(loss_D.item(), loss_G.item())
