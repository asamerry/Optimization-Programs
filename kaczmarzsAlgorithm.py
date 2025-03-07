# This program is used to find a solution to a system of linear equations with minimum norm
# i.e. solving min ||x|| s.t. Ax=b where A is mxn and m \leq n with rank(A)=m.
# Aswell, the parameter mu defines an over or undershoot of each projection between hyperplanes, 
# with exact orthogonal projection when mu=1.

import numpy as np

# Similar to normal modulus function, but returns m if m|m
def mod(k, m) -> int:
    return m if k%m == 0 else k%m

def kaczmarzsAlgorithm(A:np.ndarray[np.float64], b:np.ndarray[np.float64], iterations:int, mu:float = 1) -> None:
    # Define dimensions and initial guess of x0 = 0
    m, n = A.shape
    x = [np.zeros(n)]
    print(f"x0 = {x[0]}")

    # Iterates using Kaczmarz's algorithm
    for k in range(iterations):
        x.append( x[k] + mu * (( b[mod(k+1, m)-1] - A[mod(k+1, m)-1].T @ x[k] ) / (A[mod(k+1, m)-1].T @ A[mod(k+1, m)-1])) * A[mod(k+1, m)-1] )
        print(f"x{k+1} = {x[k+1]}")

if __name__ == "__main__":
    # Initializes A and b
    A = np.array([[1, 0, 1, 0],
                  [0, 1, 1, 1]])
    b = np.array([2, 1])

    kaczmarzsAlgorithm(A, b, iterations=5, mu=1)