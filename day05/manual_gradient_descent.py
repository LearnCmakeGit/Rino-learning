"""Day 05: Manual gradient descent with PyTorch autograd.

Goal:
- Understand training as repeated optimization.
- Manually fit y = 2x + 1.
- See the loop: forward -> loss -> backward -> update -> zero grad.
- Do NOT use nn.Module or torch.optim yet.

Run in Colab:

    python day05/manual_gradient_descent.py
"""

import torch


def main():
    # Training data generated from y = 2x + 1.
    x = torch.tensor([[0.0], [1.0], [2.0], [3.0]])
    y_true = torch.tensor([[1.0], [3.0], [5.0], [7.0]])

    # Parameters that we want to learn.
    # Start from deliberately poor guesses.
    w = torch.tensor(0.0, requires_grad=True)
    b = torch.tensor(0.0, requires_grad=True)

    learning_rate = 0.1
    num_steps = 100

    for step in range(num_steps):
        # 1. Forward: current model prediction.
        y_pred = w * x + b

        # 2. Loss: mean squared error.
        loss = torch.mean((y_pred - y_true) ** 2)

        # 3. Backward: compute d(loss)/dw and d(loss)/db.
        loss.backward()

        # Print occasionally so we can watch convergence.
        if step % 10 == 0 or step == num_steps - 1:
            print(
                f"step={step:3d}  "
                f"loss={loss.item():.6f}  "
                f"w={w.item():.6f}  "
                f"b={b.item():.6f}  "
                f"dw={w.grad.item():.6f}  "
                f"db={b.grad.item():.6f}"
            )

        # 4. Update parameters manually.
        # We do not want autograd to build a graph for the optimizer step.
        with torch.no_grad():
            w -= learning_rate * w.grad
            b -= learning_rate * b.grad

        # 5. Clear gradients because PyTorch accumulates them by default.
        w.grad.zero_()
        b.grad.zero_()

    print("\nLearned parameters:")
    print("w =", w.item())
    print("b =", b.item())

    print("\nTarget parameters:")
    print("w = 2.0")
    print("b = 1.0")

    # A quick inference test after training.
    x_test = torch.tensor([[4.0]])
    with torch.no_grad():
        y_test = w * x_test + b

    print("\nPrediction at x = 4:")
    print("y_pred =", y_test.item())
    print("expected = 9.0")


if __name__ == "__main__":
    main()
