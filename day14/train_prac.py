import torch
from torch import nn

from torch.utils.data import Dataloader, TensorDataset
from pathlib import Path
from model_prac import model_prac

def main():
    torch.manual_seed(0)
    input_u, query_x, query_u = MakeData()
    dataset = TensorDataset(input_u, query_x, query_u)
    loader = DataLoader(dataset, batch_size = 32, shuffle = True)
    
    

if __name__=="__main__":
    main()