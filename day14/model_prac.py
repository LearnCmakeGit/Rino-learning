import torch
from torch import nn


class model_prac(nn.module):
    def __init__(self, in_fea = 16, hidden_dim =64, latent_dim = 5, out_fea =16):
        