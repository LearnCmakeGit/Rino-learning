import torch
from model_prac import model_prac

from pathlib import Path

def main():
    path = Path(__file__).resolve().parent/"day13.pt"
    model = model_prac();
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    model.eval()

    input_x = torch.linspace(0,1,16).view(1,16)
    input_u = torch.sin(2*torch.pi*1.5*input_x+3.0)
    
    query_x = torch.linspace(0,1,41).view(1,41,1)
    true_u = torch.sin(2*torch.pi* 1.5*query_x + 3.0)
    
    with torch.no_grad():
        pred, z = model_prac(input_u, query)
        mse = torch.mean((pred-true_u)**2).item()
    print("input_x shape:", tuple(input_x.shape))
    print("z shape:", tuple(z.shape))
    print("query_x shape:", tuple(query_x.shape))
    print("pred shape:", tuple(pred.shape))
    print("true shape:", tuple(true_u.shape))
    print("mse :", mse)
    
    
    

if __name__ == "__main__":
    main()