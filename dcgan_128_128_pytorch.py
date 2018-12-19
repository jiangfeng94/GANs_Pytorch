import argparse
import torch,torchvision
import torch.nn as nn
from torchvision.utils import save_image

parser =argparse.ArgumentParser()
parser.add_argument('--data_path',default ='D:/Desktop/celeba')
parser.add_argument('--img_size',type=int,default=128)
parser.add_argument('--batchsize',type=int,default=128)
parser.add_argument('--netD',default='',)
parser.add_argument('--netG',default='',)
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--z_num',type=int,default=200)
opt =parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#图像读入与预处理
# 成功的使用torchvision.datasets.ImageFolder
# 这个函数的使用要求是 root路径下需要有两个文件夹，一个文件夹代表label=1，另一个代表label=0
# 假设我们不需要label 而我们的图像路径是 './root/imgpath' 我们只需要使用函数torchvision.datasets.ImageFolder('./root')

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
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator,self).__init__()
        self.dis =nn.Sequential(
            # 3×128×128
            nn.Conv2d(3,64,4,2,1,bias=False),
            nn.LeakyReLU(0.2,inplace=True),
            # 64×64×64
            nn.Conv2d(64,128,4,2,1,bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2,inplace=True),
            # 128×32×32
            nn.Conv2d(128,256,4,2,1,bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2,inplace=True),
            # 256×16×16
            nn.Conv2d(256,512,4,2,1,bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2,inplace=True),
            # 512×8×8
            nn.Conv2d(512,1024,4,2,1,bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2,inplace=True),
            # 1024×4×4
            nn.Conv2d(1024,1,4,1,0,bias=False),
            nn.Sigmoid()
        )
    def forward(self,input):
        output =self.dis(input)
        return output.view(-1,1).squeeze(1)

class Generator(nn.Module):
    def __init__(self):
        super(Generator,self).__init__()
        self.gen =nn.Sequential(
            nn.ConvTranspose2d(opt.z_num,1024,4,1,0,bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(True),
            # 1024*4*4
            nn.ConvTranspose2d(1024,512,4,2,1,bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # 512*8*8
            nn.ConvTranspose2d(512,256,4,2,1,bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # 256*16*16
            nn.ConvTranspose2d(256,128,4,2,1,bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True), 
            # 128*32*32
             nn.ConvTranspose2d(128,64,4,2,1,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True), 
            # 64*64*64
            nn.ConvTranspose2d(64,3,4,2,1,bias=False),
            nn.Tanh()
            # 3*128*128
        )
    def forward(self,input):
        output =self.gen(input)
        return output

Net_G =Generator()
Net_D =Discriminator()

Net_G.apply(weights_init)
Net_D.apply(weights_init)
#模型的读取
if opt.netD != '':
    Net_D.load_state_dict(torch.load(opt.netD))
if opt.netG !='':
    Net_G.load_state_dict(torch.load(opt.netG))

criterion = nn.BCELoss()
optimizerD = torch.optim.Adam(Net_D.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerG = torch.optim.Adam(Net_G.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

for epoch in range(200):
    for i,(imgs,label) in enumerate(dataloader):
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
             save_image(fake_imgs.data[:25], '../output_128/%d_gen.png' % (epoch * len(dataloader) + i), nrow=5, normalize=True)
             save_image(imgs.data[:25], '../output_128/%d_real.png' % (epoch * len(dataloader) + i), nrow=5, normalize=True)
    # 模型的保存
    torch.save(Net_G.state_dict(), '../save/netG_epoch_%d.pth' % (epoch))
    torch.save(Net_D.state_dict(), '../save/netD_epoch_%d.pth' % (epoch))

