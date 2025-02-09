import numpy as np
from sympy import symbols, diff, Matrix, sympify

# Get function from input
func_str = input("Enter a function f: R^n -> R ::: ")
f = sympify(func_str)

# Sets proper variables
var_names = sorted(set(filter(str.isalpha, func_str)))
vars = symbols(var_names)

# Defines the gradient of f(x)
grad_f = Matrix([diff(f, x) for x in vars])

# Search Parameters
alpha = 0.05
epsilon = 10e-25
max_iters = 500

# Initial Guess
print("Initial Guess:")
x_0 = np.array([float(input(f"x{i+1} = ")) for i, _ in enumerate(vars)])

sequence = [x_0]

for _ in range(max_iters):
    grad_f_at_x_k = np.array([grad_f[i].subs(dict(zip(vars, sequence[-1]))) for i in range(len(grad_f))], dtype=float)
    
    # Gradient Descent Algorithm: x_k+1 = x_k - alpha * grad_f(x_k)
    new_x = sequence[-1] - alpha * grad_f_at_x_k

    ### TODO: add "if ConditionsHold():" statement

    print(f"{len(sequence)}: {new_x}")
    sequence.append(new_x)
    
    # Termination condition: |f(x_k+1) - f(x_k)| < epsilon
    if abs( f.subs(dict(zip(vars, sequence[-1]))) - f.subs(dict(zip(vars, sequence[-2])))) < epsilon:
        break

    for x in sequence[-1]:
        if x == float("inf") or x == float("-inf"):
            break

print(f"\nMinimizer: x* = {sequence[-1]}")
print(f"Minimum: f(x*) = {f.subs(dict(zip(vars, sequence[-1])))}")
