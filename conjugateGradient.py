# This program is used to solve equations of the form Qx=b when Q is positive definite

import numpy as np


def conjugateGradient():
    # Displays g, d, a, and x at each step if True; displays only x if False
    showDetails = True

    # Initialize Q and b
    Q = np.array([[2, 1, 2],
                  [1, 2, 1],
                  [0, 1, 2]])
    b = np.array([1, 1, 1])

    dim = Q.shape[0]

    # Initialize lists for the iterate values -- index = subscript
    x = [np.zeros(dim)]
    g = []
    d = []
    a = []

    # Iterate the same number of times as the dimension of Q
    for k in range(dim):
        g.append( Q @ x[k] - b )
   
        # Assigning d_k requires d_k-1; the conditional statement ensures that the first iteration doesn't call d when empty ("d[-1]")
        if k == 0:
            d.append( -g[k] )
        else:
            d.append( -g[k] + ( (d[k-1].T @ (Q @ g[k])) / (d[k-1].T @ (Q @ d[k-1])) ) * d[k-1] )
        
        a.append( -(g[k].T @ d[k]) / (d[k].T @ (Q @ d[k])) )
        x.append( x[k] + a[k]*d[k] )
    
        if showDetails:
            print(f"\ng{k} = {np.round(g[k], 4)}")
            print(f"d{k} = {np.round(d[k], 4)}")
            print(f"a{k} = {np.round(a[k], 4)}")
        print(f"x{k+1} = {np.round(x[k+1], 4)}")

if __name__ == "__main__":
    conjugateGradient()
