import numpy as np
from sympy import symbols, diff, Matrix, sympify

def taylorSeries(f, vars, x_0):
    # Define the gradient and Hessian of the function
    grad_f = np.array([diff(f, var) for var in vars])
    try:
        H = np.array([[diff(g, var) for var in vars] for g in grad_f])
    except:
        return False

    # Evaluate each part at x_0
    f_x_0 = f.subs(dict(zip(vars, x_0)))
    grad_f_x_0 = np.array([grad_f_k.subs(dict(zip(vars, x_0))) for grad_f_k in grad_f])
    H_x_0 = np.array([[H_ij.subs(dict(zip(vars, x_0))) for H_ij in H_i] for H_i in H])
   
    taylorPolynomial = f_x_0 + (vars - x_0).T @ grad_f_x_0 + (1/2) * (vars - x_0).T @ H_x_0 @ (vars - x_0)

    return taylorPolynomial
