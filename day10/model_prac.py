import torch
from torch import nn

class MLP(nn.Module):
    
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        
        slef.net = nn.Sequential(
            nn.linear(in_features, hidden_features),
            nn.Relu(),
            nn.linear(hidden_features, out_features)
        )
        
    def forward(self,x):
        return self.net(x)