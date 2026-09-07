"""Day 08: Separate training and inference behavior.

Goal:
- Reuse Dataset/DataLoader training from Day07.
- Train a small nn.Linear model.
- Save learned parameters to a checkpoint.
- Switch the model to eval mode for inference.
- Use torch.no_grad() during inference.

Run in Colab:

    python day08/train_infer_split.py
"""

from pathlib import Path

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


class LineDataset(Dataset):
    def __init__(self):
        self.X = torch.tensor([
            [0.0],
            [1.0],
            [2.0],
            [3.0],
        ])
        self.Y = torch.tensor([
            [1.0],
            [3.0],
            [5.0],
            [7.0],
        ])

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.Y[index]


def train_model():
    dataset = LineDataset()
    loader = DataLoader(dataset, batch_size=2, shuffle=True)

    model = nn.Linear(in_features=1, out_features=1)
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
    print("weight =", model.weight.detach())
    print("bias   =", model.bias.detach())

    # Use the script's own directory, not the current working directory.
    # This makes checkpoint saving work even if the script is launched
    # from /content or another directory.
    script_dir = Path(__file__).resolve().parent
    checkpoint_path = script_dir / "line_model.pt"

    torch.save(model.state_dict(), checkpoint_path)
    print("\nSaved checkpoint:", checkpoint_path)

    return model


def run_inference(model):
    model.eval()

    X_test = torch.tensor([
        [4.0],
        [5.0],
    ])

    with torch.no_grad():
        Y_test = model(X_test)

    print("\nInference:")
    print("X_test =")
    print(X_test)
    print("Y_pred =")
    print(Y_test)


def main():
    model = train_model()
    run_inference(model)


if __name__ == "__main__":
    main()
