"""Day 02: NumPy matrix-vector multiplication and residual.

Goal:
- Treat NumPy arrays as small dense matrices/vectors.
- Practice A @ x.
- Practice residual r = b - A @ x.
- Connect this directly to linear solvers such as GMRES.

Run in Colab:

    python day02/numpy_matrix.py
"""

import numpy as np


def main():
    # A small 2x2 matrix.
    A = np.array([
        [4.0, 1.0],
        [2.0, 3.0],
    ])

    # A candidate solution vector x.
    x = np.array([
        [1.0],
        [2.0],
    ])

    # Right-hand side b.
    b = np.array([
        [7.0],
        [8.0],
    ])

    Ax = A @ x
    r = b - Ax

    print("A =")
    print(A)

    print("x =")
    print(x)

    print("b =")
    print(b)

    print("A @ x =")
    print(Ax)

    print("residual r = b - A @ x =")
    print(r)

    print("residual norm ||r||_2 =")
    print(np.linalg.norm(r))

    # Exact solve for comparison.
    x_exact = np.linalg.solve(A, b)
    r_exact = b - A @ x_exact

    print("exact solution x_exact =")
    print(x_exact)

    print("exact residual norm =")
    print(np.linalg.norm(r_exact))


if __name__ == "__main__":
    main()
