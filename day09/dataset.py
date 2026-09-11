"""Day 09 dataset definition."""

import torch
from torch.utils.data import Dataset


class LineDataset(Dataset):
    def __init__(self):
        self.X = torch.tensor([
            [0.0],
            [1.0],
            [2.0],
            [3.0],
        ])
        self.Y = torch.tensor([
            [1.0],
            [3.0],
            [5.0],
            [7.0],
        ])

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.Y[index]
