"""Day 07: Dataset and DataLoader.

Goal:
- Understand how training data is wrapped in a Dataset.
- Understand how DataLoader creates mini-batches.
- Inspect batch shapes and sample ordering.
- Reuse a simple nn.Linear model and optimizer.

Run in Colab:

    python day07/dataset_dataloader.py
"""

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


class LineDataset(Dataset):
    def __init__(self):
        # One sample per row.
        # X shape: [num_samples, in_features] = [8, 1]
        self.X = torch.tensor([
            [0.0],
            [1.0],
            [2.0],
            [3.0],
            [4.0],
            [5.0],
            [6.0],
            [7.0],
        ])

        # Y = 2X + 1
        # Y shape: [num_samples, out_features] = [8, 1]
        self.Y = 2.0 * self.X + 1.0

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.Y[index]


def main():
    dataset = LineDataset()

    print("dataset size =", len(dataset))
    print("first sample =", dataset[0])
    print("third sample =", dataset[2])

    batch_size = 2
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    print("\nMini-batches from one pass through DataLoader:")
    for batch_id, (X_batch, Y_batch) in enumerate(loader):
        print(f"batch {batch_id}")
        print("  X_batch.shape =", X_batch.shape)
        print("  Y_batch.shape =", Y_batch.shape)
        print("  X_batch =", X_batch.flatten().tolist())
        print("  Y_batch =", Y_batch.flatten().tolist())

    # Train the same linear model, but now one mini-batch at a time.
    model = nn.Linear(in_features=1, out_features=1)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

    num_epochs = 100

    for epoch in range(num_epochs):
        epoch_loss = 0.0

        for X_batch, Y_batch in loader:
            Y_pred = model(X_batch)
            loss = loss_fn(Y_pred, Y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # loss.item() is the mean loss for this mini-batch.
            epoch_loss += loss.item()

        if epoch % 10 == 0 or epoch == num_epochs - 1:
            mean_batch_loss = epoch_loss / len(loader)
            print(f"epoch={epoch:3d}  mean_batch_loss={mean_batch_loss:.6f}")

    print("\nLearned parameters:")
    print("weight =", model.weight.item())
    print("bias   =", model.bias.item())

    print("\nModel parameter shapes:")
    print("weight.shape =", model.weight.shape)
    print("bias.shape   =", model.bias.shape)

    X_test = torch.tensor([[8.0]])
    with torch.no_grad():
        Y_test = model(X_test)

    print("\nPrediction at X = [[8.0]]:")
    print("Y_pred   =", Y_test.item())
    print("expected = 17.0")


if __name__ == "__main__":
    main()
