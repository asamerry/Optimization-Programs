# Linear Program Solver

Both files implement the same type of problem. The MATLAB program does seem to be a bit more user friendly as it doesn't require the problem to be stated in standard equality or standard inequality form, whereas the Python algorithm does. Since every linear program can be put can be put into one of these two forms, either program can solve any linear program, however to use the Python program the user will generally need to be able to convert the program into a standard form.

The standard equality form and standard inequality form are given by $\min c^T x \text{ subject to } Ax = b, x \geq 0$ and $\min c^T x \text{ subject to } Ax \leq b, x \geq 0$ respectively. 
