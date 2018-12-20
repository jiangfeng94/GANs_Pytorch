import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
import numpy as np 
data_mean = 3.0
data_stddev = 0.2
Series_Length = 30
g_input_size = 20    
g_hidden_size = 150  
g_output_size = Series_Length
d_input_size = Series_Length
d_hidden_size = 75   
d_output_size = 1
d_minibatch_size = 15 
g_minibatch_size = 10

print_interval = 1000
d_learning_rate = 3e-3
g_learning_rate = 8e-3
def get_real_sampler(mu, sigma):
    dist = Normal( mu, sigma )
    return lambda m, n: dist.sample( (m, n) ).requires_grad_()

def get_noise_sampler():
    return lambda m, n: torch.rand(m, n).requires_grad_()  # Uniform-dist data into generator, _NOT_ Gaussian

actual_data = get_real_sampler( data_mean, data_stddev )
noise_data  = get_noise_sampler()
class Generator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Generator, self).__init__()
        self.map1 = nn.Linear(input_size, hidden_size)
        self.map2 = nn.Linear(hidden_size, hidden_size)
        self.map3 = nn.Linear(hidden_size, output_size)
        self.xfer = torch.nn.SELU()

    def forward(self, x):
        x = self.xfer( self.map1(x) )
        x = self.xfer( self.map2(x) )
        return self.xfer( self.map3( x ) )
class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Discriminator, self).__init__()
        self.map1 = nn.Linear(input_size, hidden_size)
        self.map2 = nn.Linear(hidden_size, hidden_size)
        self.map3 = nn.Linear(hidden_size, output_size)
        self.elu = torch.nn.ELU()

    def forward(self, x):
        x = self.elu(self.map1(x))
        x = self.elu(self.map2(x))
        return torch.sigmoid( self.map3(x) )
G = Generator(input_size=g_input_size, hidden_size=g_hidden_size, output_size=g_output_size)
D = Discriminator(input_size=d_input_size, hidden_size=d_hidden_size, output_size=d_output_size)
criterion = nn.BCELoss()
d_optimizer = optim.SGD(D.parameters(), lr=d_learning_rate ) #, betas=optim_betas)
g_optimizer = optim.SGD(G.parameters(), lr=g_learning_rate ) #, betas=optim_betas)
losses = []
input_data =[]  
gen_data =[]
def train_G():
    noise = noise_data( g_minibatch_size, g_input_size )
    fake_data = G( noise )
    gen_data.extend(fake_data.tolist())
    fake_decision = D( fake_data )
    error = criterion( fake_decision, torch.ones( g_minibatch_size, 1 ) )  # we want to fool, so pretend it's all genuine
    error.backward()
    return error.item(), fake_data
import matplotlib.pyplot as plt
def draw( data,path) :    
    plt.hist(data, 100)
    plt.xlabel('Heights')
    plt.ylabel('Frequency')
    plt.title('Heights Of Male Students')
    plt.savefig(path)
    plt.clf()

for epoch in range(40000):
    D.zero_grad() 
    real_data = actual_data( d_minibatch_size, d_input_size )
    decision = D( real_data )
    error = criterion( decision, torch.ones( d_minibatch_size, 1 ))  # ones = true
    error.backward()
    input_data.extend(real_data.tolist())
    noise = noise_data( d_minibatch_size, g_input_size )
    fake_data = G( noise ) 
    decision = D( fake_data )
    error = criterion( decision, torch.zeros( d_minibatch_size, 1 ))  # zeros = fake
    error.backward()
    d_optimizer.step()
    
    G.zero_grad()
    loss,generated = train_G()
    g_optimizer.step()
    
    losses.append( loss )
    if( epoch % print_interval) == (print_interval-1) :
        print( "Epoch %6d. Loss %5.3f" % ( epoch+1, loss ) )
        draw( np.array(input_data),"../output/show_plt/%d_input.jpg"%(epoch))
        draw( np.array(gen_data),"../output/show_plt/%d_gen.jpg"%(epoch))
        input_data.clear()
        gen_data.clear()
        
print( "Training complete" )


