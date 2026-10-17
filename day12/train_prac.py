import torch 
from torch import nn

from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path

from model_prac import model_prac

def MakeData(num_samples = 512, num_input_points = 16, num_query_points = 32):
    
    x_input = torch.linspace(0,1,num_input_points)
    x_query = torch.linspace(0,1,num_query_points).view(num_query_points,1)
    sample_u = []
    sample_x_query =[]
    sample_u_query=[]
    
    for i in range(num_samples):
        freq = 0.5 + 2.5 * (i/num_samples)
        phase = 0.8*torch.sin(torch.tensor(float(i)))
        
        u_input_i = torch.sin(2.0*torch.pi*freq* x_input + phase)
        u_query_i = torch.sin(2.0*torch.pi*freq* x_query[:,0]+phase)
        sample_u.append(u_input_i)
        sample_x_query.append(x_query)
        sample_u_query.append(u_query_i.view(num_query_points,1))
    u_input=torch.stack(sample_u)
    x_query=torch.stack(sample_x_query)
    u_query=torch.stack(sample_u_query)
    return u_input, x_query, u_query
        
def main():
    u_input,x_query,u_query = MakeData()
    dataset = TensorDataset(u_input, x_query, u_query)
    loader = DataLoader(dataset, batch_size = 32, shuffle = True)
    model = model_prac(num_inputs=16, hidden_dim =32,latent_dim=4)
    
    ls_fn = nn.MSELoss()
    opt = torch.optim.SGD(model.parameters(), lr =0.1)
    num_steps = 800
    model.train()
    for step in range(num_steps):
        epoch_loss = 0.0
        for batch_u_input, batch_x, batch_u in loader:
            batch_pred,z= model(batch_u_input, batch_x)
            assert batch_pred.shape == batch_u.shape,  (
                f" {batch_pred.shape} != {batch_u.shape}")
            ls = ls_fn(batch_u, batch_pred)
            opt.zero_grad()
            ls.backward()
            opt.step()
            epoch_loss += ls.item()
        if step%100 ==0 or step ==num_steps-1:
            print(f"step: {step} epoch loss: {epoch_loss}")
            
    model.eval()
    with torch.no_grad():
        samp_u = u_input[0:1]
        samp_x = x_query[0:1]
        true_u = u_query[0:1]
        u_pred,z= model(samp_u, sample_x)
    print(" input shape: ", tuple(samp_u.shape))
    print(" x query shape: ", tuple(samp_x.shape))
    print(" latent z: ", tuple(z.shape))
    print(" u query shape: ", tuple(u_pred.shape))
    
    for i in range(5):
        xp = samp_x[0,i,0].item()
        up =u_pred[0,i,0].item()
        u_true=true_u[0,i].item()
        print(f"xp: {xp:.5f}, up: {up:+.5f},true: {u_true:+.5f}")
    print("\nlatent z =", z.tolist())
    
    checkpoint = Path(__file__).resolve().parent/"check.pt"
    torch.save(model.state_dict(), checkpoint)

if __name__ == "__main__":
    main()