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
    
    model=model_prac()
    model.train()
    ls_fn = nn.MSLoss()
    opt=torch.optim.Adam(model.parameters(),lr=0.01)
    
    num_steps=800
    for step in range(num_steps):
        epoch_loss = 0.0
        for batch_u, batch_query_x, batch_query_u in loader:
            batch_pred_u, batch_z = model(batch_u,batch_query_x)
            assert(batch_pred_u.shape == batch_query_u)
            ls = ls_fn(batch_pred_u, batch_query_u)
            opt.zero_grad()
            ls.backward()
            opt.step()
            epoch_loss += ls.item()*batch_u.shape[0]
        if(step % 100 == 0 or step == num_steps-1):
            print(f"step {step:4d}, epoch_loss {epoch_loss:4f}")
    with torch.no_grad():
        samp_u = imput_u[0:1,...]
        samp_x = nn.linspace(0,1,16).view(1,16,1)
        pred_1, _= model_prac(samp_u,samp_x)
        
        
        

if __name__=="__main__":
    main()