
import torch
from torch import nn

def main():
    x = torch.tensor([[0.0],[1.0],[2.0],[3.0]])
    y_true = torch.tensor([[1.0],[3.0],[5.0],[7.0]])
    
    model = nn.Linear(in_features = 1,out_features = 1)
    
    with torch.no_grad():
        model.weight.fill_(0.0)
        model.bias.fill_(0.0)
    
    
    ls_fn = nn.MSELoss()
    
    optm = torch.optim.SDG(model.parameters(), lr=0.1)
    num_step = 100
    for i in range(num_steps):
        y = model(x)
        ls = ls_fn(y,y_true)
        optm.zero_grad()
        ls.backward()
        optm.step()
        
    print("model weight: ", model.weight.item())
    print("model bias: ", medel.bias.item())

if __name__ == "__main__":
    main()