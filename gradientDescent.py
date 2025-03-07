# Currently works for multivariate quadratics, but can easily be altered to accept other types of functions

import numpy as np

def f(x):
    return 0.5 * x.T @ Q @ x - b.T @ x

def gradientDescent(Q:np.ndarray[np.float64], b:np.ndarray[np.float64], alpha:float = 0.05, epsilon:float = 10e-25, iterations:int = 500, showDetails:bool = True) -> None:
    # Takes initial guess from input
    print("Initial Guess:")
    x0 = np.array([float(input(f"x{i+1} = ")) for i in range(Q.shape[0])])
    x = [x0]

    # Iterates a maximum of max_iters times
    for _ in range(iterations):
        grad_f_x_k = Q @ x[-1] - b

        x.append( x[-1] - alpha * grad_f_x_k )

        # Prints each iteration if true; prints only the minimum and minimizer if false
        if showDetails: 
            print(f"Iteration {len(x)}: {x[-1]}")

        # Terminated if |f(x_k+1) - f(x_k)| < epsilon or if {x_n} diverges
        if abs(f(x[-1]) - f(x[-2])) < epsilon:
            break
        for x_i in x[-1]:
            if x_i == float("inf") or x_i == float("-inf"):
                break

    print(f"\nMinimizer: x* = {x[-1]}")
    print(f"Minimum: f(x*) = {f(x[-1])}")

if __name__ == "__main__":
    # Initialize matrices Q and b
    Q = np.array([[2, -2],
                [-2, 6]])
    b = np.array([3, 0])

    gradientDescent(Q, b)