
import torch

def main():
    x=torch.tensor([
        [0.0],
        [1.0],
        [2.0],
        [3.0]
        ])
    y_true = torch.tensor([
        [1.0],
        [3.0],
        [5.0],
        [7.0]
        ])
    
    w = torch.tensor([[0.0]],requires_grad = True)
    b = torch.tensor([[0.0]],requires_grad = True)

    learn_rate = 0.25
    num_steps = 100;
    
    print("x =", x.shape)
    print("w =", w.shape)
    print("b =" ,b.shape)
    print("y = ", (x @ w.T + b).shape)
    print("y_true = ", y_true.shape)
    
    for i in range(num_steps):
        y = x@w.T + b
        loss = torch.mean((y-y_true)**2)
        loss.backward()
        
        with torch.no_grad():
            w -= learn_rate * w.grad
            b -= learn_rate * b.grad
            
        w.grad.zero_()
        b.grad.zero_()
     
    print("learn_rate = ", learn_rate)
    print("w = ", w.item())
    print("b = ", b.item())
            
    
if __name__=="__main__":
    main()



