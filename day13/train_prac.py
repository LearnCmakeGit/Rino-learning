import torch
from torch import nn

from torch.utils.data import Dataset, DataLoader
from pathlib import Path

def wave(query,freq,phas):
    u = torch.sin(2*torch.pi* freq*query + phas)
    return u

def DataGen(num_samps =512, num_inputs =16, num_queries = 24):
    input_x = torch.linspace(0,1,num_inputs).view(num_inputs,1)
    freq = 0.5 + 2.5 * torch.randm(num_samps,1)
    phas = -torch.pi + 2*torch.pi*torch.randm(num_samps,1)
    
    u_input = wave(input_x, freq, phas).squeeze(-1)
    x_query = torch.linspace(0,1,num_queries).view(num_queries,1)
    u_query = wave(x_query, freq, phas)
def main():
    u_input,x_query,u_query = DataGen()
    dataset = Dataset(u_input, x_query, u_query)
    loader = DataLoader(dataset, batch_size=32,shuffle=True)
    
    model = model_prac()
    model.train()
    ls_fn = nn.MSELoss()
    
    opt = torch.optim.SGD(model.parameters(),lr =0.1)
    num_steps = 800
    
    for step in range(num_steps):
        epoch_loss = 0
        for batch_u_inp, batch_x_query, batch_u_query in loader:
            batch_u_pred,z=model(batch_u_inp,batch_x_query)
            ls = ls_fn(batch_u_query, batch_u_pred)
            opt.zero_grad()
            ls.backward()
            opt.step()
            epoch_loss += ls.item()*batch_u_input.shape(0)
        if step %100 == 0 or step == nums_step-1:
            print(f"step {step:3d}, epoch_loss: {epoch_loss / len(loader):4f}")
    model.eval()
    with torch.no_grad():
        samp0 = u_input[0:1]
        #check1  predict on arbitrarily locs
        new_x = torch.linspace(0,1,101).view(1,101,1)
        pred101,z0 = model(samp0,new_x)
        
        #check2 mse for samp0 on input_x locations
        query_x0 = torch.linspace(0,1,16).view(1,16,1)
        pred0,_ = model(samp0,query_x0)
        mse = ls_fn(samp0,pred0(...,0)).item()
        print("samp0 shape", tuple(samp0.shape))
        print("z0 shape", tuple(z0.shape))
        print("pred0 shape", tuple(pred0.shape))
        print(f"mse, {mse:6f}")
        
    checkpoint=Path(__file__).resolve().parent/"day13.pt"
    torch.save(model.state_dict(),checkpoint)
        
if __name__ == "__main__":
    main()