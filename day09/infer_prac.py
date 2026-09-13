from pathlib import Path
import torch
from torch import nn

from model_prac import model_prac

def main():
    model = model_prac()
    checkpoint = Path(__file__).resolve().parent / "checkpoint.pt"
    state = torch.load(checkpoint, map_location = "cpu")
    model.load_state_dict(state)
    
    model.eval()
    
    test_x = torch.tensor([[10.0],[12.0]])
    with torch.no_grad():
        y_pred = model(test_x)

    print("predicted: ", y_pred.tolist())

if __name__ == "__main__":
    main()