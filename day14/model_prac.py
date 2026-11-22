import torch
from torch import nn


class model_prac(nn.module):
    def __init__(self, in_fea = 16, hidden_dim =64, latent_dim = 5, out_fea =24):
        super().__init__()
        self.encoder = nn.Sequential(
          nn.Linear(in_fea, hidden_dim),
          nn.ReLU(),
          nn.Linear(hidden_dim, latent_dim)
          
        )
        self.decoder = nn.Sequential(
          nn.Linear(latent_dim +1, hidden_dim)
          nn.tanh()
          nn.Linear(hidden_dim, 1)
        
        
        )