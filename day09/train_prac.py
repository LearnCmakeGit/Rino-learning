
from pathlib import Path
import torch
from torch.nn.data import DataLoader

from data_prac import LineData
from model_prac import model_prac

def main():
    dataset = LineData()
    dataloader = DataLoader(dataset, batch_size=2, shuffle = True)
    
    model = model_prac()
    
    ls_fn = torch.MSELoss()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    
    num_steps = 100
    for step in range(num_steps):
        epoch_loss = 0.0
        for (patch_x,patch_y) in dataloader:
            y = model(patch_x)
            ls = ls_fn(y,patch_y)
            opt.zero_grad()
            ls.backward()
            opt.step()
        epoch_loss += ls.item()
        if step % 10 == 0 or step == num_steps-1:
            print(f"step {step:3d}: epoch loss {epoch_loss:.4e}")
    
    print(" save check point")
    checkpoint = Path(__file__).resolve().parent()/"checkpoint.pt"
    torch.save(torch.state_dict(),checkpoint)



if __name__ == "__main__":
    main()