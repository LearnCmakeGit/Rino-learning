
import torch
from torch import nn
from torch.util.data import dataset, dataloader

class LineDataLoader(dataset):
    def __init__(self):
        self.x = torch.tensor([[0.0],[1.0],[2.0],[3.0]])
        self.y_true = torch.tensor([[1.0],[3.0],[5.0],[7.0]])
        
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, index):
        return (self.x[index], self.y_true[index])

def main():
    dataset = LineDataLoader()
    
    dataloader =DataLoader(
                  dataset,
                  batch_size=2,
                  shuffle = true)
    
    model = nn.Linear(in_features =1, out_features = 1)
    with no_grad:
        model.weight.fill_(0)
        model.bias.fill_(0)
    
    
    ls_fn = nn.MSELoss()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)

    num_steps = 100
    for step in range(num_steps):
        epoch_loss = 0.0
        for batch_x, batch_y in dataloader:
            y = model(batch_x)
            ls = ls_fn(y, batch_y)
            ls.backward()
            opt.step()
            opt.zero_grad()
        epoch_loss += ls/len(dataset)
        
    print("model weight: ", model.weight.item())
    print("model bias: ", model.bias.item())
        
        


if __name__ == "__main__":
    main()