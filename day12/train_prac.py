import torch 
from torch import nn

from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path

from model_prac import model_prac

def MakeData():
    u_input = nn.Linearspace(0,1,16)
    x_query = nn.Linearspace(0,1,16).view(1,16)
    
    
def main():
    u_input,x_query,u_query = MakeData()
    dataset = TensorDataset(u_input, x_query, u_query)
    loader = DataLoader(dataset, batch_size = 32, shuffle = True)
    model = model_prac(in_features=16, hidden_dim =32,latent_dim=32)
    
    ls_fn = nn.MSELoss()
    opt = torch.optim.SGD(model.parameters(), lr =0.1)
    num_steps = 800
    model.train()
    for step in range(num_steps):
        epoch_loss = 0.0
        for batch_u_input, batch_x, batch_u in loader:
            batch_pred,z= model(batch_u_input, batch_x)
            ls = ls_fn(batch_u, batch_pred)
            opt.zero_grad()
            ls.backward()
            opt.step()
            epoch_loss += ls.item()
        if step%100 ==0 or step ==num_steps-1:
            print(f"step: "{step} " epoch loss: "{epoch_loss})
            
    model.eval()
    with torch.no_grad():
        samp_u = u_input[0:1]
        samp_x = x_query[0:1]
        true_u = u_query[0:1]
        u_pred,z= model(u_input, x_query)
    print(" input shape: ", tuple(samp_u.shape))
    print(" x query shape: ", tuple(samp_x.shape))
    print(" latent z: ", tuple(z.shape))
    print(" u query shape: " tuple(u_pred))
    
    for i in range(5):
        xp = samp_x[0,i,0].item()
        up =u_pred[0,i,0].item()
        u_true=true_u[0,i,0].item()
        print(f"xp: "{xp:.5f}, "up: "{up:+.5f},"true: "{u_true:+.5f})
    
    
    checkpoint = Path(__file__).resolve().parent/"check.pt"
    torch.save(model.dict_state(), checkpoint)

if __name__ == "__main__":
    main()