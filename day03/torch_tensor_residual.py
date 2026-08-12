"""Day 03: PyTorch tensor basics.

Goal:
- Reproduce the Day02 NumPy residual calculation with torch.Tensor.
- Compare ndarray and Tensor syntax.
- Inspect dtype, shape, device, and storage-sharing behavior.
- Prepare for autograd and GPU use later.

Run in Colab:

    python day03/torch_tensor_residual.py
"""

import torch


def show_tensor(name, x):
    print(f"{name} =")
    print(x)
    print(f"shape={tuple(x.shape)}, dtype={x.dtype}, device={x.device}")
    print()


def main():
    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print()

    A = torch.tensor([
        [4.0, 1.0],
        [2.0, 3.0],
    ], dtype=torch.float64)

    x = torch.tensor([
        [1.0],
        [2.0],
    ], dtype=torch.float64)

    b = torch.tensor([
        [7.0],
        [8.0],
    ], dtype=torch.float64)

    Ax = A @ x
    r = b - Ax

    show_tensor("A", A)
    show_tensor("x", x)
    show_tensor("b", b)
    show_tensor("A @ x", Ax)
    show_tensor("r = b - A @ x", r)

    print("Residual 2-norm:", torch.linalg.vector_norm(r).item())
    print()

    x_exact = torch.linalg.solve(A, b)
    r_exact = b - A @ x_exact

    show_tensor("x_exact", x_exact)
    print("Exact residual norm:", torch.linalg.vector_norm(r_exact).item())
    print()

    print("Reference / clone experiment")
    print("----------------------------")

    t1 = torch.tensor([1.0, 2.0, 3.0])
    t2 = t1
    t3 = t1.clone()

    print("Initial:")
    print("t1 =", t1)
    print("t2 =", t2)
    print("t3 =", t3)
    print()

    t2[0] = 99.0

    print("After t2[0] = 99:")
    print("t1 =", t1, "  <- changed because t2 = t1 shares the same object/storage")
    print("t2 =", t2)
    print("t3 =", t3, "  <- unchanged because clone() copied the tensor")


if __name__ == "__main__":
    main()
