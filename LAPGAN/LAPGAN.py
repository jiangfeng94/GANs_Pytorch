import torch
import torch.nn as nn

class Sub_D(nn.Module):
    def __init__(self,n_layer):
        self.n_layer =n_layer
        
        for layer in range (self.n_layer):
            if layer == (self.n_layer - 1):
                n_conv_in = 3
            else:
                


class LAPGAN(object):
    def __init__(self,n_level):
        self.n_level =n_level
        self.D_Nets =[]
        self.G_Nets =[]
        self.z_dim =10
        for level in range(self.n_level):
            cur_layer =n_level-level
            if level ==(n_level -1):
                condtion =False
            else:
                condtion =True
            Net_D =Sub_D()
            Net_G =Sub_G()
            Net_D.cuda()
            Net_G.cuda()
            self.D_Nets.append(Net_D)
            self.G_Nets.append(Net_G)
    def train(self,batchsize,get_level=None, generator=False):
        self.outputs=[]
        self.generator_outputs =[]
        FloatTensor =torch.cuda.FloatTensor
        for level in range(self.n_level):
            G_Net =self.G_Nets[self.n_level-level-1]
            z =Variable(FloatTensor(np.random.uniform(low=-1.0, high=1.0,size=(batchsize, self.z_dim)))) 
            if level =0:
                output_imgs= G_Net(z)
                output_imgs = output_imgs.data.numpy()
                self.generator_outputs.append(output_imgs)
            else:
                input_imgs =np.array([[cv2.pyrUp(output_imgs[i, j, :])
                                      for j in range(3)]
                                      for i in range(batchsize)])
                condition_imgs =Variable(FloatTensor(input_imgs))
                residual_imgs = G_Net(z, condition_imgs)
                output_imgs = residual_imgs.data.numpy() + input_imgs
                self.generator_outputs.append(residual_imgs.data.numpy())
            self.outputs.append(output_imgs)
        if get_level is None:
            get_level = -1
        if generator:
            result_imgs = self.generator_outputs[get_level]
        else:
            result_imgs = self.outputs[get_level]
        return result_imgs
