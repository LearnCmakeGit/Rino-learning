import torch
from torch import nn

class model_prac(nmodules):
    def __init__(self, num_inputs = 16, num_query=32,hidden_dim= 32, latent_dim=4):
        super():__init__()
        
        self.encoder = nn.sequential(
            nn.linear(num_input, hidden_dim),
            nn.ReLU()
            nn.linear(hidden_dim, latent_dim)
        )
        
        self.decoder = nn.sequential(
            nn.linear(latent_dim+1, hidden_dim),
            nn.ReLU()
            nn.linear(hidden_dim, hidden_dim)
            nn.ReLU()
            nn.linear(hidden_dim,1)
            )
            
    def encode(self,u_input):
        return self.encoder(u_input)
        
        
    def decod(self, x_query,z):
        batch_size, num_query, num_coor = x_query.shape()
    
    def forward(self, u_input,x_query):
        z=self.encoder(u_input)
        u_pred = self.decoder(x_query,z)
        return u_pred, z
      