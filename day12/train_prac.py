import torch 
from torch import nn

from torch.utils.data import DataLoader, TensorData
from pathlib import Path

from model_prac import model_prac

def MakeData():
    
    
def main():
    u_input,x_query,u_query = MakeData()
    dataset = TensorData(u_input, x_query, u_query)
    loader = DataLoader(dataset, batch_size = 32, shuffle = True)
    model = model_prac(in_features=16, hiden_dim =4, out_features=32)
    
    ls_fn = nn.MSELoss()
    opt = torch.optim.SGD(model.parameters(), lr =0.1)
    
    
    

if __name__ == "__main__":
    main()