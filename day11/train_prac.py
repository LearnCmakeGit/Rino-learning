import torch
from torch import nn
from torch.utils.data import TensorDataset DataLoader

from model_prac import AutoEncoder

def main():
    X = dataPrep()
    
    data = TensorDataset(X,X)
    dataloader = DataLoader(data, batch_size=32, shuffle = True)
    
    model = AutoEncoder(in_features = 16, latent_dim = 3, out_features=16)
    ls_fn = nn.MSELoss()
    opt = torch.optim.SGD(model.parameters(), lt=0.1)
    model.train()
    
    num_steps = 1000
    for step in range(num_steps):
        epoch_loss = 0.0
        for batch_x, batch_y in dataloader:
            y_pred =model(batch_x)
            ls = ls_fn(y_pred, batch_y)
             opt.zero_grad()
            ls.backward()
            opt.step()
            epoch_loss += ls.item()
        if(step % 100==0 or step == nums_steps -1):
            prinf(f"epoch_loss: {epoch_loss}")
            
    print(model)
    for name, param in model.parameters():
        print(")
    
            
            
    
    

if __name__ == "__main__":
    main()