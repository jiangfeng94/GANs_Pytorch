import torch
import torch.nn as nn

class Sub_D(nn.Module):

    def __init__(self, n_layer=3, condition=True, n_condition=100,
                 use_gpu=False, featmap_dim=256, n_channel=1,
                 condi_featmap_dim=256):
        """
        Conditional Discriminator.
        Architecture brought from DCGAN.
        """
        super(CondiGAN_Discriminator, self).__init__()
        self.n_layer = n_layer
        self.condition = condition

        # original Discriminator
        self.featmap_dim = featmap_dim

        convs = []
        BNs = []
        for layer in range(self.n_layer):
            if layer == (self.n_layer - 1):
                n_conv_in = n_channel
            else:
                n_conv_in = int(featmap_dim / (2**(layer + 1)))
            n_conv_out = int(featmap_dim / (2**layer))

            _conv = nn.Conv2d(n_conv_in, n_conv_out, kernel_size=5,
                              stride=2, padding=2)
            if use_gpu:
                _conv = _conv.cuda()
            convs.append(_conv)

            if layer != (self.n_layer - 1):
                _BN = nn.BatchNorm2d(n_conv_out)
                if use_gpu:
                    _BN = _BN.cuda()
                BNs.append(_BN)

        # extra image information to be conditioned on
        if self.condition:
            self.condi_featmap_dim = condi_featmap_dim
            convs_condi = []
            BNs_condi = []

            for layer in range(self.n_layer):
                if layer == (self.n_layer - 1):
                    n_conv_in = n_channel
                else:
                    n_conv_in = int(condi_featmap_dim / (2**(layer + 1)))
                n_conv_out = int(condi_featmap_dim / (2**layer))

                _conv = nn.Conv2d(n_conv_in, n_conv_out, kernel_size=5,
                                  stride=2, padding=2)
                if use_gpu:
                    _conv = _conv.cuda()
                convs_condi.append(_conv)

                if layer != (self.n_layer - 1):
                    _BN = nn.BatchNorm2d(n_conv_out)
                    if use_gpu:
                        _BN = _BN.cuda()
                    BNs_condi.append(_BN)

            self.fc_c = nn.Linear(condi_featmap_dim * 4 * 4, n_condition)

        # register layer modules
        self.convs = nn.ModuleList(convs)
        self.BNs = nn.ModuleList(BNs)
        if self.condition:
            self.convs_condi = nn.ModuleList(convs_condi)
            self.BNs_condi = nn.ModuleList(BNs_condi)

        # output layer
        n_hidden = featmap_dim * 4 * 4
        if self.condition:
            n_hidden += n_condition
        self.fc = nn.Linear(n_hidden, 1)

    def forward(self, x, condi_x=None):
        """
        Concatenate CNN-processed extra information vector at the last layer
        """
        for layer in range(self.n_layer):
            conv_layer = self.convs[self.n_layer - layer - 1]
            if layer == 0:
                x = F.leaky_relu(conv_layer(x), negative_slope=0.2)
            else:
                BN_layer = self.BNs[self.n_layer - layer - 1]
                x = F.leaky_relu(BN_layer(conv_layer(x)), negative_slope=0.2)
        x = x.view(-1, self.featmap_dim * 4 * 4)

        # calculate and concatenate extra information
        if self.condition:
            for layer in range(self.n_layer):
                _conv = self.convs_condi[self.n_layer - layer - 1]
                if layer == 0:
                    condi_x = F.leaky_relu(_conv(condi_x), negative_slope=0.2)
                else:
                    BN_layer = self.BNs_condi[self.n_layer - layer - 1]
                    condi_x = F.leaky_relu(BN_layer(_conv(condi_x)),
                                           negative_slope=0.2)

            condi_x = condi_x.view(-1, self.condi_featmap_dim * 4 * 4)
            condi_x = self.fc_c(condi_x)
            x = torch.cat((x, condi_x), 1)

        # output layer
        x = F.sigmoid(self.fc(x))

        return x


class Sub_G(nn.Module):

    def __init__(self, noise_dim=10, n_layer=3, condition=True,
                 n_condition=100, use_gpu=False, featmap_dim=256, n_channel=1,
                 condi_featmap_dim=256):
        """
        Conditional Generator.
        Architecture brought from DCGAN.
        """
        super(CondiGAN_Generator, self).__init__()
        self.n_layer = n_layer
        self.condition = condition

        # extra image information to be conditioned on
        if self.condition:
            self.condi_featmap_dim = condi_featmap_dim

            convs_condi = []
            BNs_condi = []
            for layer in range(self.n_layer):
                if layer == (self.n_layer - 1):
                    n_conv_in = n_channel
                else:
                    n_conv_in = int(condi_featmap_dim / (2**(layer + 1)))
                n_conv_out = int(condi_featmap_dim / (2**layer))

                _conv = nn.Conv2d(n_conv_in, n_conv_out, kernel_size=5,
                                  stride=2, padding=2)
                if use_gpu:
                    _conv = _conv.cuda()
                convs_condi.append(_conv)

                if layer != (self.n_layer - 1):
                    _BN = nn.BatchNorm2d(n_conv_out)
                    if use_gpu:
                        _BN = _BN.cuda()
                    BNs_condi.append(_BN)

            self.fc_c = nn.Linear(condi_featmap_dim * 4 * 4, n_condition)

        # calculate input dimension
        n_input = noise_dim
        if self.condition:
            n_input += n_condition

        # Generator
        self.featmap_dim = featmap_dim
        self.fc1 = nn.Linear(n_input, int(featmap_dim * 4 * 4))

        convs = []
        BNs = []
        for layer in range(self.n_layer):
            if layer == 0:
                n_conv_out = n_channel
            else:
                n_conv_out = featmap_dim / (2 ** (self.n_layer - layer))
            n_conv_in = featmap_dim / (2 ** (self.n_layer - layer - 1))

            n_width = 5 if layer == (self.n_layer - 1) else 6
            _conv = nn.ConvTranspose2d(n_conv_in, n_conv_out, n_width,
                                       stride=2, padding=2)
            if use_gpu:
                _conv = _conv.cuda()
            convs.append(_conv)

            if layer != 0:
                _BN = nn.BatchNorm2d(n_conv_out)
                if use_gpu:
                    _BN = _BN.cuda()
                BNs.append(_BN)

        # register layer modules
        self.convs = nn.ModuleList(convs)
        self.BNs = nn.ModuleList(BNs)
        if self.condition:
            self.convs_condi = nn.ModuleList(convs_condi)
            self.BNs_condi = nn.ModuleList(BNs_condi)

    def forward(self, x, condi_x=None):
        """
        Concatenate CNN-processed extra information vector at the first layer
        """
        # calculate and concatenate extra information
        if self.condition:
            for layer in range(self.n_layer):
                _conv = self.convs_condi[self.n_layer - layer - 1]
                if layer == 0:
                    condi_x = F.leaky_relu(_conv(condi_x), negative_slope=0.2)
                else:
                    BN_layer = self.BNs_condi[self.n_layer - layer - 1]
                    condi_x = F.leaky_relu(BN_layer(_conv(condi_x)),
                                           negative_slope=0.2)

            condi_x = condi_x.view(-1, self.condi_featmap_dim * 4 * 4)
            condi_x = self.fc_c(condi_x)
            x = torch.cat((x, condi_x), 1)

        x = self.fc1(x)
        x = x.view(-1, self.featmap_dim, 4, 4)

        for layer in range(self.n_layer):
            conv_layer = self.convs[self.n_layer - layer - 1]
            if layer == (self.n_layer - 1):
                x = F.tanh(conv_layer(x))
            else:
                BN_layer = self.BNs[self.n_layer - layer - 2]
                x = F.relu(BN_layer(conv_layer(x)))

        return x
                


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
