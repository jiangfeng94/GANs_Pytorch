# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 20:29:44 2018

@author: SalaFeng-
"""

import torch
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
## DATASET
train_dataset = datasets.MNIST(root='./mnist_data/',train=True,transform=transforms.ToTensor(),download=True)
test_dataset = datasets.MNIST(root='./mnist_data/',train=False,transform=transforms.ToTensor())
train_loader =torch.utils.data.DataLoader(dataset=train_dataset,batch_size=64,shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=64,shuffle=False)
## NETWORK
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.l1 = nn.Linear(784,520)
        self.l2 = nn.Linear(520,320)
        self.l3 = nn.Linear(320,240)
        self.l4 = nn.Linear(240,120)
        self.l5 = nn.Linear(120,10)
    
    def forward(self,x):
        x = x.view(-1,784)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        x =F.log_softmax(self.l5(x))
        return x
model =Net()
optimizer =optim.SGD(model.parameters(),lr=0.01,momentum=0.5)
def train(epoch):
    for batch_idx,(data,target) in enumerate(train_loader):
        data,target=Variable(data),Variable(target)
        optimizer.zero_grad()
        output=model(data)
        loss=F.nll_loss(output,target)
        loss.backward()
        optimizer.step()
        if batch_idx %100 ==0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data), len(train_loader.dataset),100. * batch_idx / len(train_loader), loss.data))
def test():
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.nll_loss(output, target).data
        pred = output.data.max(1)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, len(test_loader.dataset),100. * correct / len(test_loader.dataset)))
for epoch in range(30):
    train(epoch)
    test()