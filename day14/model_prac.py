import torch
from torch import nn


class model_prac(nn.Module):
    def __init__(self, in_fea = 16, hidden_dim =64, latent_dim = 5):
        super().__init__()
        self.encoder = nn.Sequential(
          nn.Linear(in_fea, hidden_dim),
          nn.Tanh(),
          nn.Linear(hidden_dim, latent_dim)
        )
        self.decoder = nn.Sequential(
          nn.Linear(latent_dim +1, hidden_dim),
          nn.Tanh(),
          nn.Linear(hidden_dim, hidden_dim),
          nn.Tanh(),
          nn.Linear(hidden_dim, 1)
        )
    def encode(self, input_u):
        return self.encoder(input_u)
    def decode(self, z, query_x):
        assert(z.shape[0]==query_x.shape[0])
        z_repeat=z.unsqueze(1).expand(-1,query_x.shape[1],-1)
        decode_in = torch.cat(z_repeat,query_x,dim=-1)
        return self.decoder(decode_in)
    def forward(self, input_u, 