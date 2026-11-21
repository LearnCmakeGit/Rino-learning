import torch
from model_prac import model_prac

from pathlib import Path

def main():
    path = Path(__file__).resolve().parent/"day13.pt"
    model = model_prac();
    model.load_state_dict(torch.load(path, position=cpu, only_weights=True)
    model.eval()


if __name__ == "__main__":
    main()