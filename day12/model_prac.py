import torch
from torch import nn

class model_prac(nmodules):
    def __init__(self, num_inputs = 16, num_query=32,hidden_dim= 32, latent_dim=4):
        super():__init__()
        
        encode = nn.sequential(
            nn.linear(num_input, hidden_dim),
            nn.ReLU()
            nn.linear(hidden_dim, latent_dim)
        )
        
        decode=nn.sequential(
            nn.linear(latent_dim+1, hidden_dim),
            nn.ReLU()
            nn.linear(hidden_dim, num_query)
            )
            
    def encoder(self,u_input):
        return encode(u_input)
        
        
    def decoder(self, x_query,z):
        
    
    def forward(self, u_input,x_query):
        z=encoder(u_input)
        u_pred = decoder(x_query,z)
        return u_pred, z
      