import torch
from torch import nn

class model_prac(nn.Module):
    
    def __init__(self, in_fea = 16, hidden_dim = 8, latent_dim = 4):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_fea, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, latent_dim)
        )
        self.decoder =nn.Sequential(
            nn.Linear(latent_dim+1, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim,1)
        )
    def encode(self, u_input):
        return self.encoder(u_input)
    def decode(self, z, x_query):
        assert z.dim() == 2
        z_repeat = z.unsqueeze(1).expand(-1,x_query.shape[1],-1)
        decode_input = torch.cat((z_repeat,x_query), dim=-1)
        return self.decoder(decode_input)
    def forward(self, u_input, x_query):
        z=self.encode(u_input)
        return self.decode(z, x_query), z
        
    