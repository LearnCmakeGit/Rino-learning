import torch
from torch import nn

from torch.utils.data import Dataset, DataLoader
from pathlib import Path



def main():
    u_input,x_query,u_query = DataGen()
    dataset = Dataset(u_input, x_query, u_query)
    loader = DataLoader(dataset, batch_size=32,shuffle=True)
    
    model = model_prac()
    
    ls_fn = nn.MSELoss()
    
    opt = torch.optim.SGD(model.parameters(),lr =0.1)
    num_steps = 800
    
    for stepcin range(num_steps):
        epoch_loss = 0
        for batch_u_inp, batch_x_query, batch_u_query in loader:
            batch_u_pred,z=model(batch_u_inp,batch_x_query)
            ls = ls_fn(batch_u_query, batch_u_pred)
            opt.zero_grad()
            ls.backward()
            opt.step()
            epoch_loss += ls.item()
    
        
    

if __name__ == "__main__":
    main()