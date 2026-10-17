import torch
from torch import nn


class QueryFunctionModel(nn.Module):
    """Encode 16 function samples into z, then evaluate u(x_query)."""

    def __init__(self, num_input_points=16, hidden_dim=64, latent_dim=8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(num_input_points, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + 1, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def encode(self, u_input):
        return self.encoder(u_input)

    def decode(self, z, x_query):
        # z: [batch, latent_dim], x_query: [batch, queries, 1]
        assert z.ndim == 2
        assert x_query.ndim == 3 and x_query.shape[-1] == 1
        assert z.shape[0] == x_query.shape[0]
        z_repeated = z.unsqueeze(1).expand(-1, x_query.shape[1], -1)
        decoder_input = torch.cat((z_repeated, x_query), dim=-1)
        return self.decoder(decoder_input)

    def forward(self, u_input, x_query):
        z = self.encode(u_input)
        return self.decode(z, x_query), z
