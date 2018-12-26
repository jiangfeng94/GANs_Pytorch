import torch
import torch.nn as nn
class Generator(nn.Module):
    def __init__(self,img_shape=[3,128,128]):
        super(Generator,self).__init__()
        self.shape =img_shape
        self.init_size =self.shape[1] // 4
        self.fc =nn.Sequential(nn.Linear(100,128 *self.init_size**2 ))
        self.G_0 = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128,128,3,stride=1,padding=1),
            nn.BatchNorm2d(128,0.8),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Upsample(scale_factor=2)
        )
        self.G_A =nn.Sequential(
            nn.Conv2d(128,64,3,stride=1,padding=1),
            nn.BatchNorm2d(64,0.8),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(64,self.shape[0],3,stride=1,padding=1),
            nn.Tanh()
        )
        self.G_B =nn.Sequential(
            nn.Conv2d(128,64,3,stride=1,padding=1),
            nn.BatchNorm2d(64,0.8),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(64,self.shape[0],3,stride=1,padding=1),
            nn.Tanh()
        )
    def forward(self,z):
        out =self.fc(z)
        out = out.view(out.shape[0],128,self.init_size,self.init_size)
        tmp =self.G_0(out)
        img_A =self.G_A(tmp)
        img_B =self.G_B(tmp)
        return img_A,img_B

class Discriminator(nn.Module):
    def __init__(self,img_shape=[3,128,128]):
        super(Discriminator,self).__init__()
        def block(in_,out_,normalize=True):
            out =[nn.Conv2d(in_,out_,img_shape[0],2,1)]
            if normalize:
                out.append(nn.BatchNorm2d(out_,0.8))
            out.extend([nn.LeakyReLU(0.2,inplace=True),nn.Dropout2d(0.25)])
            return out
        ds_size = img_shape[1] // 2**4
        self.D_A =nn.Linear(128*ds_size**2,1)
        self.D_B =nn.Linear(128*ds_size**2,1)
        self.D_shared =nn.Sequential(
            *block(3,16,normalize=False),
            *block(16,32),
            *block(32,64),
            *block(64,128)
        )
    def forward(self,img_A,img_B):
        d_A =self.D_shared(img_A)
        d_A =d_A.view(d_A.shape[0],-1)
        valid_A= self.D_A(d_A)
        d_B =self.D_shared(img_B)
        d_B =d_B.view(d_B.shape[0],-1)
        valid_B= self.D_B(d_B)
        return valid_A,valid_B