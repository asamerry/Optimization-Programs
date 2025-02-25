from scipy.optimize import linprog

# Coefficients for the objective function (cost per serving)
c = [0.14, 0.12, 0.2, 0.75, 0.15]

# Coefficients for the inequality constraints (nutritional values per serving)
A = [
    [-23, -171, -65, -112, -188],  # Calories (>= 2000)
    [-0.1, -0.2, 0, -9.3, -16],    # Fat (>= 50)
    [-0.6, -3.7, -2.2, -7, -7.7],  # Protein (>= 100)
    [-6, -30, -13, 0, -2]          # Carbohydrates (>= 250)
]

# Right-hand side of the constraints
b = [-2000, -50, -100, -250]

# Bounds for the decision variables (non-negativity)
x_bounds = [(0, None)] * 5

# Solve the LP
result = linprog(c, A_ub=A, b_ub=b, bounds=x_bounds, method='highs')

# Output the result
print("Optimal servings:", result.x)
print("Minimum cost:", result.fun)
