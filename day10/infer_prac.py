from pathlib import Path
import torch
from model_prac import MLP


def main():
    model = MLP(in_features=1, hidden_features=8, out_features=1)
    
    checkpoint= Path(__file__).resolve().parent/"mlp.pt"
    state = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state)
    
    test = torch.tensor([[10.0],[15.0]])
    with torch.no_grad():
        y_pred = model(test)
    
    print("pred y: ", y_pred.tolist())
    


if __name__ == "__main__":
    main()