import torch
from torch import nn


class QueryDecoderAutoEncoder(nn.Module):
    """
    Encode a sampled function u(x_i) into a latent vector z,
    then decode the value u(x_q) at arbitrary query coordinates x_q.
    """

    def __init__(self, num_input_points=16, latent_dim=4, hidden_dim=32):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(num_input_points, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

        # decoder input = [z, x_query]
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def encode(self, u_samples):
        return self.encoder(u_samples)

    def decode(self, z, x_query):
        # z:       [batch, latent_dim]
        # x_query: [batch, num_queries, 1]
        batch_size, num_queries, _ = x_query.shape

        # Repeat the latent vector so every query point sees the same z.
        z_expanded = z.unsqueeze(1).expand(-1, num_queries, -1)

        decoder_input = torch.cat([z_expanded, x_query], dim=-1)
        return self.decoder(decoder_input)

    def forward(self, u_samples, x_query):
        z = self.encode(u_samples)
        y_query = self.decode(z, x_query)
        return y_query, z
