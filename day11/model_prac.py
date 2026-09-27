import torch

from torch import nn

class AutoEncoder(nn.module):
    def __init__(self, in_features = 16, 
                 latent_dim = 3,out_features=16):
    super():__init__()
    self.encoder = nn.Sequential{
                    nn.linear(in_features, letent_dim)
                    nn.RELU()
                    nn.linear(latent_dim, latent_dim)}
    self.decoder = nn.Sequential{
                   nn.linear(latent_dim, latent_dim)
                   nn.RELU()
                   nn.linear(latent_dim, out_features)
                   }
    def encoder(self,x):
        return self.encoder(x)
    def decoder(self,z):
        return self.decoder(z)
    def forward(self, x):
        z=encoder(x)
        recon=decoder(z)
        return recon