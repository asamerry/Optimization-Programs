import numpy as np
from sympy import diff, Matrix, sympify
from sys import exit


def newtonsMethod(f, vars, epsilon:float = 10e-25, max_iters:int = 500, display:bool = True) -> None:
 
    # Initial Guess
    print("Initial Guess:")
    x_0 = np.array([float(input(f"{var} = ")) for var in vars])

    # Defines the gradient of f(x)
    grad_f = np.array([diff(f, x) for x in vars])

    # Defines the Hessain of f(x)
    H = np.array([[diff(grad_f_k, x) for x in vars] for grad_f_k in grad_f])

    sequence = [x_0]

    for _ in range(max_iters):
        grad_f_x_k = np.array([grad_f_k.subs(dict(zip(vars, sequence[-1])))] for grad_f_k in grad_f)
        H_x_k = np.array([[H_ij.subs(dict(zip(vars, sequence[-1]))) for H_ij in H_i] for H_i in H])
        
        if np.linalg.det(H_x_k) != 0:
            new_x = sequence[-1] - np.linalg.inv(H_x_k) @ grad_f_x_k
        else:
            print("Non-singular Hessian, using constant step size.")
            exit(1)

        if display:
            print(f"Iteration {len(sequence)}: {new_x}")
        
        sequence.append(new_x)
    
        # Terminates if |f(x_k+1) - f(x_k)| < epsilon
        if abs(float(f.subs(dict(zip(vars, sequence[-1])))) - float(f.subs(dict(zip(vars, sequence[-2]))))) < epsilon:
            break

        # Terminates if sequence diverges
        for x in sequence[-1]:
            if x == float("inf") or x == float("-inf"):
                break

    print(f"\nMinimizer: x* = {sequence[-1]}")
    print(f"Minimum: f(x*) = {f.subs(dict(zip(vars, sequence[-1])))}")
