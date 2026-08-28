"""Day 05: Manual gradient descent with PyTorch autograd.

Goal:
- Understand training as repeated optimization.
- Use the common ML convention: one sample per row.
- Fit Y = X @ W + b to data generated from y = 2x + 1.
- See the loop: forward -> loss -> backward -> update -> zero grad.
- Do NOT use nn.Module or torch.optim yet.

Run in Colab:

    python day05/manual_gradient_descent.py
"""

import torch


def main():
    # Four samples, one feature per sample.
    # Shape: [num_samples, in_features] = [4, 1]
    X = torch.tensor([
        [0.0],
        [1.0],
        [2.0],
        [3.0],
    ])

    # Target output for each sample.
    # Shape: [num_samples, out_features] = [4, 1]
    Y_true = torch.tensor([
        [1.0],
        [3.0],
        [5.0],
        [7.0],
    ])

    # Parameters to learn.
    # W shape: [in_features, out_features] = [1, 1]
    # b shape: [out_features] = [1]
    W = torch.tensor([[0.0]], requires_grad=True)
    b = torch.tensor([0.0], requires_grad=True)

    print("X.shape =", X.shape)
    print("W.shape =", W.shape)
    print("b.shape =", b.shape)
    print("Y_true.shape =", Y_true.shape)

    learning_rate = 0.1
    num_steps = 100

    for step in range(num_steps):
        # 1. Forward: common row-sample convention.
        # Y_pred shape: [4, 1]
        Y_pred = X @ W + b

        # 2. Loss: mean squared error.
        loss = torch.mean((Y_pred - Y_true) ** 2)

        # 3. Backward: compute d(loss)/dW and d(loss)/db.
        loss.backward()

        if step % 10 == 0 or step == num_steps - 1:
            print(
                f"step={step:3d}  "
                f"loss={loss.item():.6f}  "
                f"W={W.item():.6f}  "
                f"b={b.item():.6f}  "
                f"dW={W.grad.item():.6f}  "
                f"db={b.grad.item():.6f}"
            )

        # 4. Manual gradient-descent update.
        with torch.no_grad():
            W -= learning_rate * W.grad
            b -= learning_rate * b.grad

        # 5. Clear accumulated gradients.
        W.grad.zero_()
        b.grad.zero_()

    print("\nLearned parameters:")
    print("W =", W.item())
    print("b =", b.item())

    print("\nTarget parameters:")
    print("W = 2.0")
    print("b = 1.0")

    # Inference: still one sample per row.
    X_test = torch.tensor([[4.0]])  # shape [1, 1]
    with torch.no_grad():
        Y_test = X_test @ W + b

    print("\nPrediction at X = [[4.0]]:")
    print("Y_pred =", Y_test.item())
    print("expected = 9.0")


if __name__ == "__main__":
    main()
