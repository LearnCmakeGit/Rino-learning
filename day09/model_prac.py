import torch
from torch import nn

class model_prac(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(in_features = 1, out_features = 1)
    
    def forward(self, x):
        return self.lin(x)