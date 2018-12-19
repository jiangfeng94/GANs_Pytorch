import argparse
import torch,torchvision
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import  datasets,transforms
from torchvision.utils import save_image
parser = argparse.ArgumentParser()
parser.add_argument('--batchsize', type=int, default=256, help='size of the batches')
parser.add_argument('--lr', type=float, default=0.0002, help='adam: learning rate')
parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
parser.add_argument('--latent_dim', type=int, default=100, help='dimensionality of the latent space')
parser.add_argument('--n_classes', type=int, default=10, help='number of classes for dataset')
parser.add_argument('--img_size', type=int, default=32, help='size of each image dimension')
parser.add_argument('--channels', type=int, default=1, help='number of image channels')
opt = parser.parse_args()

dataset = datasets.MNIST('../../mnist_data', train=True, download=True,
                   transform=transforms.Compose([
                        transforms.Resize(opt.img_size),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                   ]))
dataloader = DataLoader(dataset,batch_size=opt.batchsize, drop_last=True,shuffle=True)

class Generator(nn.Module):
    def __init__(self):
        super(Generator,self).__init__()
        self.label_emb =nn.Embedding(opt.n_classes,opt.n_classes)
        def conv_block(in_feat,out_feat,normalize=True):
            layers =[nn.Linear(in_feat,out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat,0.8))
            layers.append(nn.LeakyReLU(0.2,inplace=True))
            return layers
        self.gen =nn.Sequential(
            *conv_block(opt.latent_dim+opt.n_classes, 128, normalize=False),
            *conv_block(128,256),
            *conv_block(256,512),
            *conv_block(512,1024),
            nn.Linear(1024,int(np.prod([opt.channels,opt.img_size,opt.img_size]))),
            nn.Tanh()
        )
    def forward(self,noise,labels):
        gen_input =torch.cat((self.label_emb(labels),noise),-1)
        img =self.gen(gen_input)
        img =img.view(img.size(0),opt.channels,opt.img_size,opt.img_size)
        return img
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator,self).__init__()
        self.label_emb = nn.Embedding(opt.n_classes, opt.n_classes)
        self.dis = nn.Sequential(
            nn.Linear(opt.n_classes + int(np.prod((opt.channels,opt.img_size,opt.img_size))), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1)
        )
    def forward(self,img,labels):
        img =img.view(img.size(0),-1)
        labels =self.label_emb(labels)
        d_in =torch.cat((img,labels),-1)
        return self.dis(d_in)

adversarial_loss = torch.nn.MSELoss()
auxiliary_loss = torch.nn.CrossEntropyLoss()

generator =Generator()
discriminator =Discriminator()

#cuda
discriminator.cuda()
generator.cuda()
adversarial_loss.cuda()
auxiliary_loss.cuda()
FloatTensor = torch.cuda.FloatTensor
LongTensor = torch.cuda.LongTensor
#optim
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
# Train
for epoch in range(100):
    for i, (imgs, labels) in enumerate(dataloader):

        batch_size = imgs.shape[0]
        valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
        fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)
        real_imgs = Variable(imgs.type(FloatTensor))
        labels = Variable(labels.type(LongTensor))
        optimizer_G.zero_grad()

        z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, opt.latent_dim))))
        gen_labels = Variable(LongTensor(np.random.randint(0, opt.n_classes, batch_size)))
        gen_imgs = generator(z, gen_labels)
        validity = discriminator(gen_imgs, gen_labels)
        g_loss = adversarial_loss(validity, valid)

        g_loss.backward()
        optimizer_G.step()

        optimizer_D.zero_grad()
        validity_real = discriminator(real_imgs, labels)
        d_real_loss = adversarial_loss(validity_real, valid)
        validity_fake = discriminator(gen_imgs.detach(), gen_labels)
        d_fake_loss = adversarial_loss(validity_fake, fake)

        d_loss = (d_real_loss + d_fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

        print ("[Epoch %d] [Batch %d/%d] [D loss: %f] [G loss: %f]" % (epoch,  i, len(dataloader),
                                                            d_loss.item(), g_loss.item()))
        batches_done = epoch * len(dataloader) + i

        if batches_done % 100 == 0:
            save_image(gen_imgs.data[:100], '../../mnist_cgan/%d.png' % (epoch * len(dataloader) + i), nrow=10, normalize=True)