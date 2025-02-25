# This program is used to find a solution to a system of linear equations with minimum norm
# i.e. solving min ||x|| s.t. Ax=b where A is mxn and m \leq n with rank(A)=m

import numpy as np


def mod(k, m):
    if k%m == 0:
        return m
    return k%m

def kaczmarzsAlgorithm():
    # Initializes A and B
    A = np.array([[1, 0, 1, 0],
                  [0, 1, 1, 1]])
    b = np.array([2, 1])
    
    # Sets number of iterations and value of mu for over or undershooting the actual projection
    iters = 5
    mu = 1

    m, n = A.shape
    x = [np.zeros(n)]
    print(f"x0 = {x[0]}")

    for k in range(iters):
        x.append( x[k] + mu * (( b[mod(k+1, m)-1] - A[mod(k+1, m)-1].T @ x[k] ) / (A[mod(k+1, m)-1].T @ A[mod(k+1, m)-1])) * A[mod(k+1, m)-1] )
        print(f"x{k+1} = {x[k+1]}")
    

if __name__ == "__main__":
    kaczmarzsAlgorithm()
