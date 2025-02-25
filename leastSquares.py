# This program is used to solve equations of the form min ||Ax-b|| where x* = (A^TA)\invA^Tb

import numpy as np


def main():
    A = np.array([[1, 0, 1, 0],
                  [0, 1, 1, 1]])
    b = np.array([2, 1])

    opt_x = np.linalg.inv(A.T @ A) @ A.T @ b

    print(f"x* = {opt_x}")

if __name__ == "__main__":
    main()
