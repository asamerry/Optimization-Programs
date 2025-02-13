import numpy as np
from sympy import diff, Matrix, sympify


def gradientDescent(f, vars, epsilon:float = 10e-25, max_iters:int = 500, display:bool = True) -> None:
 
    # Initial Guess
    print("Initial Guess:")
    x_0 = np.array([float(input(f"{var} = ")) for var in vars])

    # Defines the gradient of f(x)
    grad_f = Matrix([diff(f, x) for x in vars])

    # Defines the Hessain of f(x)
    H = Matrix([[diff(f, x) for x in vars] for f in grad_f])

    sequence = [x_0]

    for _ in range(max_iters):
        grad_f_at_x_k = np.array([grad_f[i].subs(dict(zip(vars, sequence[-1]))) for i in range(len(grad_f))], dtype=np.float64)

        H_x_k = H.subs(dict(zip(vars, sequence[-1])))

        if H_x_k.det() != 0:
            alpha = H_x_k.inv()

            # Gradient Descent Algorithm: x_k+1 = x_k - alpha * grad_f(x_k)
            new_x = sequence[-1] - alpha @ grad_f_at_x_k
        
        else:
            print("Non-singular Hessian, using constant step size.")
            
            alpha = 0.05
            
            new_x = sequence[-1] - alpha * grad_f_at_x_k
        
    
        ### TODO: add "if ConditionsHold():" statement

        if display:
            print(f"Iteration {len(sequence)}: {new_x}")
        
        sequence.append(new_x)
    
        # Termination condition: |f(x_k+1) - f(x_k)| < epsilon
        if abs(f.subs(dict(zip(vars, sequence[-1]))) - f.subs(dict(zip(vars, sequence[-2])))) < epsilon:
            break

        for x in sequence[-1]:
            if x == float("inf") or x == float("-inf"):
                break

    print(f"\nMinimizer: x* = {sequence[-1]}")
    print(f"Minimum: f(x*) = {f.subs(dict(zip(vars, sequence[-1])))}")
