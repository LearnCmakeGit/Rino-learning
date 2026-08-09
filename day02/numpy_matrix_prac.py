

import numpy as np

main():
    A = np.array([[1,2],[3,4]])
    b = np.array([1,2])
    
    x = np.array([0,0])
    r = b-A@x
    print ("residual: ", np.linalg(r))
    



if __name__ = "__main__":
    main()