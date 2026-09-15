from pathlib import Path
import torch
from torch import nn

def main():
    x = torch.linspace(-2,2,41).reshape(-1,1)
    y = x**2
    
    model = MLP(in_features = 1, hidden_features = 8, out_features = 1)
    ls_fn = nn.MSELoss()
    opt = torch.optim.SGD(model.parameters(),lr=0.1)
    model.train()
    
    num_steps =1000
    for step in range(num_steps):
        y_pred = model(x)
        ls = ls_fn(y,y_pred)
        opt.zero_grad()
        ls.backward()
        opt.step()
        if step %100 == 0 or step == num_steps-1:
            print(f"step {step:3d}, loss: {ls.item():.4e}")
    checkpoint = Path(__file__).resolve().parent/"mlp.pt"
    torch.save(model.state_dict(),checkpoint)
    
        
        




if __name__ == "__main__":
    main()