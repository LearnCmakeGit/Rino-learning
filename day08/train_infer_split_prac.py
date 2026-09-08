from pathlib import Path
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

class LineDataset(Dataset):
    def __init__(self):
        self.x = torch.tensor([[0.0],[1.0],[2.0],[3.0]])
        self.y_true = 2.0 * self.x + 1.0
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, index):
        return (self.x[index], self.y_true[index])
    
def model_train():
    dataset = LineDataset()
    dataloader = DataLoader(
                 dataset,
                 batch_size = 2,
                 shuffle = True)
    
    model = nn.Linear(in_features = 1, out_features = 1)
    model.train()
    with torch.no_grad():
        model.weight.fill_(0.0)
        model.bias.fill_(0.0)
    
    ls_fn = nn.MSELoss()
    opt = torch.optim.SGD(model.parameters(), lr = 0.1)
    
    num_steps = 100
    for step in range(num_steps):
        epoch_loss = 0.0
        for (batch_x, batch_yt) in dataloader:
            y = model(batch_x)
            ls = ls_fn(y,batch_yt)
            opt.zero_grad()
            ls.backward()
            opt.step()
        epoch_loss += ls.item()
        if step % 10 == 0 or step == num_steps -1:
            print (f"epoch: {step:3d} , epoch_loss: {epoch_loss:.4e}")
    print(" Train complete")
    return model
    
def model_infer(model):
    model.eval()
    x_test = torch.tensor([[10.0], [11.0]])
    with torch.no_grad():
        y_test = model(x_test)
    print("x_test: ", x_test.tolist())
    print("y_prod: ", y_test.tolist())

def main():
    model = model_train()
    model_infer(model)
    
    

if __name__ == "__main__":
    main()