from pathlib import Path

import torch

from model import MLP


def main():
    model = MLP(in_features=1, hidden_features=8, out_features=1)

    checkpoint = Path(__file__).resolve().parent / "mlp_x2.pt"
    state = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    X_test = torch.tensor([
        [-1.5],
        [-0.5],
        [0.5],
        [1.5],
    ])

    with torch.no_grad():
        Y_pred = model(X_test)

    print("X_test =")
    print(X_test)

    print("\nY_pred =")
    print(Y_pred)

    print("\nExact X^2 =")
    print(X_test ** 2)


if __name__ == "__main__":
    main()
