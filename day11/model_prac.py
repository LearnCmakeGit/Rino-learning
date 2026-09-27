import torch

from torch import nn

class AutoEncoder(nn.Module):
    def __init__(self, in_features = 16, 
                 latent_dim = 3,out_features=16):
        super().__init__()
        self.encoder = nn.Sequential(
                    nn.Linear(in_features, 8),
                    nn.ReLU(),
                    nn.Linear(8,latent_dim))
        self.decoder = nn.Sequential(
                   nn.Linear(latent_dim,8),
                   nn.ReLU(),
                   nn.Linear(8,out_features)
                   )
    def encode(self,x):
        return self.encoder(x)
    def decode(self,z):
        return self.decoder(z)
    def forward(self, x):
        z=self.encoder(x)
        recon=self.decoder(z)
        return recon