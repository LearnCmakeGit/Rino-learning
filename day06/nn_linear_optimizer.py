"""Day 06: Replace manual parameters/updates with nn.Linear and torch.optim.

Goal:
- Understand nn.Module at a minimal level.
- Inspect nn.Linear weight/bias shapes.
- Use the same row-sample convention as Day05.
- Replace manual gradient-descent updates with torch.optim.SGD.
- See that the training loop structure is still the same.

Run in Colab:

    python day06/nn_linear_optimizer.py
"""

import torch
from torch import nn


def main():
    # Four samples, one input feature per sample.
    # Shape: [batch, in_features] = [4, 1]
    X = torch.tensor([
        [0.0],
        [1.0],
        [2.0],
        [3.0],
    ])

    # Targets generated from y = 2x + 1.
    # Shape: [batch, out_features] = [4, 1]
    Y_true = torch.tensor([
        [1.0],
        [3.0],
        [5.0],
        [7.0],
    ])

    # nn.Linear(in_features, out_features)
    # Internally:
    #   weight shape = [out_features, in_features]
    #   bias shape   = [out_features]
    # Forward is conceptually:
    #   Y = X @ weight.T + bias
    model = nn.Linear(in_features=1, out_features=1)

    # Start from exactly the same initial guess as Day05.
    with torch.no_grad():
        model.weight.fill_(0.0)
        model.bias.fill_(0.0)

    print("X.shape =", X.shape)
    print("model.weight.shape =", model.weight.shape)
    print("model.bias.shape =", model.bias.shape)
    print("Y_true.shape =", Y_true.shape)

    # PyTorch loss object. Equivalent to:
    # torch.mean((Y_pred - Y_true) ** 2)
    loss_fn = nn.MSELoss()

    # Optimizer owns the parameter update rule.
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    num_steps = 100

    for step in range(num_steps):
        # 1. Forward.
        # Calling model(X) invokes nn.Linear.forward(X).
        Y_pred = model(X)

        # 2. Loss.
        loss = loss_fn(Y_pred, Y_true)

        # 3. Clear old gradients BEFORE backward.
        optimizer.zero_grad()

        # 4. Backward.
        loss.backward()

        if step % 10 == 0 or step == num_steps - 1:
            print(
                f"step={step:3d}  "
                f"loss={loss.item():.6f}  "
                f"W={model.weight.item():.6f}  "
                f"b={model.bias.item():.6f}  "
                f"dW={model.weight.grad.item():.6f}  "
                f"db={model.bias.grad.item():.6f}"
            )

        # 5. Parameter update.
        optimizer.step()

    print("\nLearned parameters:")
    print("W =", model.weight.item())
    print("b =", model.bias.item())

    print("\nTarget parameters:")
    print("W = 2.0")
    print("b = 1.0")

    # Inference after training.
    X_test = torch.tensor([[4.0]])
    with torch.no_grad():
        Y_test = model(X_test)

    print("\nPrediction at X = [[4.0]]:")
    print("Y_pred =", Y_test.item())
    print("expected = 9.0")


if __name__ == "__main__":
    main()
