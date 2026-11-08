import torch
from torch import nn

class model_prac(nn.modules):
    super().__init__()
    def __init__(self, in_fea = 16, hidden_dim = 8, latent_dim = 4):
        self.encoder = nn.Sequential(
         nn.Linear(in_fea, hidden_dim),
         nn.Tanh(),
         nn.Linear(hidden_dim, latent_dim)
         nn.Tanh())
        self.decoder =nn.Sequential(
      
      
       )
    