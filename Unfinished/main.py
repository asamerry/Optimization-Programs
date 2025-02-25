import numpy as np
from sympy import symbols, sympify
import sys

from gradientDescent import gradientDescent
from taylorSeries import taylorSeries


def getParameters():
    
    # Get function from imput and convert to a useable state
    func_str = "x**2 + 3*y**2 - 2*x*y - 3*x + 2" #input("Enter a function f:R^n -> R ::: ")
    f = sympify(func_str)

    # Get the variables from the given function
    var_names = sorted(set(filter(str.isalpha, func_str)))
    vars = symbols(var_names)

    # Get initial guess
    # x_0 = input("Inital guess: ")
    x_0 = np.array([0, 0])

    return f, vars, x_0


def main() -> None:
   
    # Get parameters for optimization
    f, vars, x_0 = getParameters()

    # Find the Taylor Polynomial of the given function
    approx_f = taylorSeries(f, vars, x_0)
    if not approx_f:
        print("Hessian uncomputable")
        sys.exit(1)

    # gradientDescent(f, vars, epsilion = 10e-25, max_iters = 500, display=True)
    gradientDescent(approx_f, vars)
    
if __name__ == "__main__": 
    main()
