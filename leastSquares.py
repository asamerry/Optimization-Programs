# This program is used to solve equations of the form min ||Ax-b|| where x* = (A^TA)\invA^Tb

import numpy as np


def leastSquares():
    A = np.array([[1, 2],
                  [1, 1],
                  [3, 2]])
    b = np.array([6, 4, 5])

    opt_x = np.linalg.inv(A.T @ A) @ A.T @ b

    print(f"x* = {opt_x}")

if __name__ == "__main__":
    leastSquares()
