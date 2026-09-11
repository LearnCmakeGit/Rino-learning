"""Day 09 training script.

Run from the repository root:

    python day09/train.py
"""

from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset import LineDataset
from model import LineModel


def main():
    dataset = LineDataset()
    loader = DataLoader(dataset, batch_size=2, shuffle=True)

    model = LineModel()
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    model.train()

    for epoch in range(100):
        epoch_loss = 0.0

        for X_batch, Y_batch in loader:
            Y_pred = model(X_batch)
            loss = loss_fn(Y_pred, Y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        if epoch % 20 == 0 or epoch == 99:
            print(f"epoch={epoch:3d} loss={epoch_loss:.6f}")

    print("\nLearned parameters:")
    print("weight =", model.linear.weight.detach())
    print("bias   =", model.linear.bias.detach())

    checkpoint_path = Path(__file__).resolve().parent / "line_model.pt"
    torch.save(model.state_dict(), checkpoint_path)
    print("\nSaved checkpoint:", checkpoint_path)


if __name__ == "__main__":
    main()
