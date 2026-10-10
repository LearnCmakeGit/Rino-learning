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
    num_steps = 800
    
    for step in range(num_steps):
        epoch_loss = 0.0
        for batch_u_input, batch_x, batch_u in loader:
            batch_pred = model(batch_u_input, batch_x)
            ls = ls_fn(batch_u, batch_pred)
            opt.zero_grad()
            
    

if __name__ == "__main__":
    main()