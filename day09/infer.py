"""Day 09 inference script.

Run from the repository root after training:

    python day09/infer.py
"""

from pathlib import Path

import torch

from model import LineModel


def main():
    checkpoint_path = Path(__file__).resolve().parent / "line_model.pt"

    model = LineModel()
    state = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    X_test = torch.tensor([
        [4.0],
        [5.0],
    ])

    with torch.no_grad():
        Y_pred = model(X_test)

    print("X_test =")
    print(X_test)
    print("Y_pred =")
    print(Y_pred)


if __name__ == "__main__":
    main()
