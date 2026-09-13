from pathlib import Path

import torch
from torch import nn

from model import MLP


def main():
    X = torch.linspace(-2.0, 2.0, 41).reshape(-1, 1)
    Y = X ** 2

    model = MLP(in_features=1, hidden_features=8, out_features=1)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    model.train()

    for epoch in range(2000):
        Y_pred = model(X)
        loss = loss_fn(Y_pred, Y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 400 == 0 or epoch == 1999:
            print(f"epoch={epoch:4d} loss={loss.item():.6f}")

    print("\nModel:")
    print(model)

    print("\nParameter shapes:")
    for name, param in model.named_parameters():
        print(f"{name:20s} shape={tuple(param.shape)}")

    checkpoint = Path(__file__).resolve().parent / "mlp_x2.pt"
    torch.save(model.state_dict(), checkpoint)
    print("\nSaved:", checkpoint)


if __name__ == "__main__":
    main()
