import torch

def main():
    x = tensor(3, requires_grad = true)
    y = x**2 +2 * x + 1
    y.backward()
    print("x = ", x)
    print("dy/dx = ", x.grad())
    

if __name__ == "__main__":
    main()