import torch.nn as nn
import torch

class ResidualBlock(nn.Module):
    def __init__(self,in_features):
        super(ResidualBlock,self).__init__()
        conv_block=[nn.ReflectionPad2d(1),
                    nn.Conv2d(in_features,in_features,3),
                    nn.InstanceNorm2d(in_features),
                    nn.ReLU(inplace=True),
                    nn.ReflectionPad2d(1),
                    nn.Conv2d(in_features,in_features,3),
                    nn.InstanceNorm2d(in_features)
        ]
        self.conv_block =nn.Sequential(*conv_block)
    def forward(self,x):
        return x +self.conv_block(x)

class Generator(nn.Module):
    def __init__(self,in_channels=3,outchannels=3,res_blocks=9):
        super(Generator,self).__init__()
        model  = [nn.ReflectionPad2d(3),
                  nn.Conv2d(3,64,7),
                  nn.InstanceNorm2d(64),
                  nn.ReLU(inplace=True)
                ]
        model += [nn.Conv2d(64,128,3,stride=2,padding=1),
                  nn.InstanceNorm2d(128),
                   nn.ReLU(inplace=True)]
        #[256,64,64]
        model += [nn.Conv2d(128,256,3,stride=2,padding=1),
                  nn.InstanceNorm2d(256),
                   nn.ReLU(inplace=True)]
        #[256,64,64]
        for _ in range(res_blocks):
            model += [ResidualBlock(256)]
        
        
        model += [nn.ConvTranspose2d(256,128,3,stride=2,padding=1,output_padding=1),
                  nn.InstanceNorm2d(128),
                   nn.ReLU(inplace=True)]
                
        model += [nn.ConvTranspose2d(128,64,3,stride=2,padding=1,output_padding=1),
                  nn.InstanceNorm2d(64),
                   nn.ReLU(inplace=True)]
        model += [nn.ReflectionPad2d(3),
                  nn.Conv2d(64,3,7),
                  nn.Tanh()]
        self.gen =nn.Sequential(*model)
    def forward(self,x):
        return self.gen(x)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator,self).__init__()
        def block(in_features,out_features,normlize=True):
            layers =[nn.Conv2d(in_features,out_features,4,stride=2,padding=1)]
            if normlize:
                layers.append(nn.InstanceNorm2d(out_features))
            layers.append(nn.LeakyReLU(0.2,inplace=True))
            return layers
        self.dis =nn.Sequential(
            *block(3,64,normlize=False),
            *block(64,128),
            *block(128,256),
            *block(256,512),
            #nn.ZeroPad2d 零填充边界 四个参数分别是左右上下
            nn.ZeroPad2d((1,0,1,0)),
            nn.Conv2d(512,1,4,padding=1)
        )
    def forward(self,img):
        return self.dis(img)
            