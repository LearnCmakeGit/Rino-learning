import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, in_features=1, hidden_features=8, out_features=1):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, out_features),
        )

    def forward(self, x):
        return self.net(x)
