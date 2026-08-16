"""Day 04: PyTorch autograd from a scalar function.

Goal:
- Understand requires_grad=True.
- Build a computation graph through ordinary tensor operations.
- Call backward() to compute derivatives automatically.
- Compare autograd with the analytical derivative.

Run in Colab:

    python day04/autograd_scalar.py
"""

import torch


def main():
    # Start with one scalar tensor.
    # requires_grad=True tells PyTorch to track operations on x
    # so that derivatives can later be computed.
    x = torch.tensor(3.0, requires_grad=True)

    # Define a simple scalar function:
    # y = x^2 + 2x + 1
    y = x**2 + 2.0 * x + 1.0

    print("x =", x)
    print("y =", y)
    print("x.requires_grad =", x.requires_grad)
    print("y.grad_fn =", y.grad_fn)

    # Compute dy/dx automatically.
    y.backward()

    print("autograd dy/dx =", x.grad)

    # Analytical derivative:
    # dy/dx = 2x + 2
    exact_grad = 2.0 * x.detach() + 2.0
    print("analytical dy/dx =", exact_grad)

    # Important: gradients accumulate by default.
    # Calling backward() again adds another gradient contribution.
    y2 = x**2
    y2.backward()
    print("after another backward(), accumulated x.grad =", x.grad)

    # Reset the stored gradient.
    x.grad.zero_()
    print("after x.grad.zero_(), x.grad =", x.grad)


if __name__ == "__main__":
    main()
