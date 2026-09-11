"""Day 09 model definition."""

from torch import nn


class LineModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(in_features=1, out_features=1)

    def forward(self, X):
        return self.linear(X)
