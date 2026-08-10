

import numpy as np

def main():
    A = np.array([[1,2],[3,4]])
    b = np.array([1,2])
    print("A shape: ", A.shape)
    print("b shape: ", b.shape)
    x = np.array([0,0])
    r = b-A@x
    print ("residual: ", np.linalg.norm(r))
    



if __name__ == "__main__":
    main()