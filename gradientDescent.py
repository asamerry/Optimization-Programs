import numpy as np
from sympy import diff, Matrix, sympify


def gradientDescent(f, vars, alpha:float = 0.05, epsilon:float = 10e-25, max_iters:int = 500, display:bool = True) -> None:
 
    # Initial Guess
    print("Initial Guess:")
    x_0 = np.array([float(input(f"{var} = ")) for var in vars])

    # Defines the gradient of f(x)
    grad_f = Matrix([diff(f, x) for x in vars])

    # Defines the Hessian of f(x)
    #H = Matrix([[diff(f, x) for x in vars] for f in grad_f])

    sequence = [x_0]

    for _ in range(max_iters):
        # Evaluates derivatives at x_k
        grad_f_x_k = np.array([grad_f_k.subs(dict(zip(vars, sequence[-1]))) for grad_f_k in grad_f])
        #H_x_k = np.array([[H_ij.subs(dict(zip(vars, sequence[-1]))) for H_ij in H_i] for H_i in H])

        # Gradient Descent Algorithm: x_k+1 = x_k - alpha * grad_f_x_k
        new_x = sequence[-1] - alpha * grad_f_x_k
        
        if display:
            print(f"Iteration {len(sequence)}: {new_x}")
        
        sequence.append(new_x)
    
        # Terminates if |f(x_k+1) - f(x_k)| < epsilon
        if abs(f.subs(dict(zip(vars, sequence[-1]))) - f.subs(dict(zip(vars, sequence[-2])))) < epsilon:
            break

        # Terminates if sequence diverges
        for x in sequence[-1]:
            if x == float("inf") or x == float("-inf"):
                break

    print(f"\nMinimizer: x* = {sequence[-1]}")
    print(f"Minimum: f(x*) = {f.subs(dict(zip(vars, sequence[-1])))}")
