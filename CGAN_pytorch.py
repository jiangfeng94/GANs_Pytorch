import torch,torchvision
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from torch.utils.data import DataLoader
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator,self).__init__()
        self.dis =nn.Sequential(
            nn.Conv2d(1,32,5,stride=1,padding=2),
            nn.LeakyReLU(0.2,True),
            nn.MaxPool2d((2,2)),
            nn.Conv2d(32,64,5,stride=1,padding=2),
            nn.LeakyReLU(0.2,True),
            nn.MaxPool2d((2,2))
        )
        self.fc =nn.Sequential(
            nn.Linear(7*7*64,1024),
            nn.LeakyReLU(0.2,True),
            nn.Linear(1024,10),
            nn.Sigmoid()
        )
    def forward(self,x):
        x =self.dis(x)
        x =x.view(x.size(0),-1)
        x =self.fc(x)
        return x

class Generator(nn.Module):
    def __init__(self,input_size,num_feature):
        super(Generator,self).__init__()
        self.fc =nn.Linear(input_size,num_feature)
        self.br =nn.Sequential(
            nn.BatchNorm2d(1),
            nn.ReLU(True)
        )
        self.gen =nn.Sequential(
            nn.Conv2d(1,50,3,stride=1,padding=1),
            nn.BatchNorm2d(50),
            nn.ReLU(True),
            nn.Conv2d(50,25,3,stride=1,padding=1),
            nn.BatchNorm2d(35),
            nn.ReLU(True),
            nn.Conv2d(25,1,3,stride=2),
            nn.Tanh()
        )
    def forward(self,x):
        x =self.fc(x)
        x =x.view(x.size(0),1,56,56)
        x=self.br(x)
        x =self.gen(x)
        return x
trans_img=torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
trainset=torchvision.datasets.MNIST('../mnist_data',train=True,transform=trans_img,download=True)
trainloader=DataLoader(trainset,batch_size=128,shuffle=True,num_workers=10)

Net_D =Discriminator()
Net_G =Generator(100,1*56*56)
criterion = nn.BCELoss()
Net_D =Net_D.cuda()
Net_G =Net_G.cuda()

d_optimizer=torch.optim.Adam(Net_D.parameters(),lr=0.0003)
g_optimizer=torch.optim.Adam(Net_G.parameters(),lr=0.0003)
for epoch in range(100):
    for i,(img,label) in enumerate(trainloader):
        print(label)
        labels_onehot = np.zeros((128,10))
        labels_onehot[np.arange(128),label.numpy()]=1
        real_label=Variable(torch.from_numpy(labels_onehot).float()).cuda()
        fake_label=Variable(torch.zeros(128,10)).cuda()
        #D
        img =Variable(img).cuda()
        real_d =Net_D(img)
        d_loss_real =criterion(real_d,real_label)
        z=Variable(torch.randn(128,100)).cuda()
        fake_img =Net_G(z)
        fake_d =Net_D(fake_img)
        d_loss_fake =criterion(fake_d,fake_label)
        d_loss=d_loss_real+d_loss_fake
        d_optimizer.zero_grad() #判别器D的梯度归零
        d_loss.backward() #反向传播
        d_optimizer.step()
        for k in range(2):
            z=torch.randn(128,100)
            z=np.concatenate((z.numpy(),labels_onehot),axis=1)
            z=Variable(torch.from_numpy(z).float()).cuda()
            print(z)
            g_loss =criterion(Net_D(Net_G(z)),real_label)
            g_optimizer.zero_grad() #生成器G的梯度归零
            g_loss.backward() #反向传播
            g_optimizer.step()

