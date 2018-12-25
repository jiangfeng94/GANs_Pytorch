import glob
import random
import os
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import torch
import sys
from torch.autograd import Variable
import numpy as np

from torchvision.utils import save_image
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)
class ImageDataset_night2day(Dataset):
    def __init__(self,root, unaligned=False,transforms_=None, mode='train'):
        self.transform = transforms.Compose(transforms_)
        self.files = sorted(glob.glob(os.path.join(root,'%s'%mode)+'/*.*'))
    def __getitem__(self,index):
        file_path = self.files[index]
        img_all = Image.open(file_path).convert('RGB')
        w =img_all.size[0]//2
        h =img_all.size[1]
        img_A =img_all.crop((0, 0, w, h))
        img_B =img_all.crop((w, 0, w*2, h))
        item_A = self.transform(img_A)
        item_B = self.transform(img_B)
        return {'A': item_A, 'B': item_B}
    def __len__(self):
        return len(self.files)



class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False, mode='train'):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned

        self.files_A = sorted(glob.glob(os.path.join(root, '%sA' % mode) + '/*.*'))
        self.files_B = sorted(glob.glob(os.path.join(root, '%sB' % mode) + '/*.*'))

    def __getitem__(self, index):
        item_A = self.transform(Image.open(self.files_A[index % len(self.files_A)]))

        if self.unaligned:
            item_B = self.transform(Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)]))
        else:
            item_B = self.transform(Image.open(self.files_B[index % len(self.files_B)]))

        return {'A': item_A, 'B': item_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))