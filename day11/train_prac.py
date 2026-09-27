import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path
from model_prac import AutoEncoder

def dataPrep(num_samples=256,num_points=16):
    x=torch.linspace(0,1,num_points)
    sample=[]
    for i in range (num_samples):
        a = 0.5 + 2.5*(i/num_samples)
        phase = torch.sin(torch.tensor(float(i)))
        u=torch.sin(2.0*torch.pi*a*x +phase)
        sample.append(u)
    X = torch.stack(sample)
    return X

def main():
    X = dataPrep()
    
    data = TensorDataset(X,X)
    dataloader = DataLoader(data, batch_size=32, shuffle = True)
    
    model = AutoEncoder(in_features = 16, latent_dim = 3, out_features=16)
    ls_fn = nn.MSELoss()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
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
        if(step % 100==0 or step == num_steps -1):
            print(f"epoch_loss: {epoch_loss}")
            
    print(model)
    for name, param in model.parameters():
        print(f"{name:25s}, {tuple(param.shape)}")
    
    model.eval()
    with torch.no_grad():
        sample = X[0:1]
        z=model.encoder(sample)
        recon = model.decoder(z)
        
    print("latent space: ", z.tolist())
    
    checkpoint = Path(__file__).resolve().parent/"auto.pt"
    torch.save(model.state_dict(),checkpoint)
            
            
    
    

if __name__ == "__main__":
    main()