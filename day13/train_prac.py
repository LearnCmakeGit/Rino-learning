import torch
from torch import nn

from torch.utils.data import Dataset, DataLoader
from pathlib import Path



def main():
    u_input,x_query,u_query = DataGen()
    dataset = Dataset(u_input, x_query, u_query)
    loader = DataLoader(dataset, batch_size=32,shuffle=True)
    
    

if __name__ == "__main__":
    main()