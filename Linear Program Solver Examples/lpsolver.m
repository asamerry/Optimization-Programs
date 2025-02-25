clear

x1 = optimvar('x1', 'LowerBound', 0, 'UpperBound', Inf);
x2 = optimvar('x2', 'LowerBound', 0, 'UpperBound', Inf);
x3 = optimvar('x3', 'LowerBound', 0, 'UpperBound', Inf);
x4 = optimvar('x4', 'LowerBound', 0, 'UpperBound', Inf);
x5 = optimvar('x5', 'LowerBound', 0, 'UpperBound', Inf);

prob = optimproblem('Objective', 0.14*x1 + 0.12*x2 + 0.2*x3 + 0.75*x4 + 0.15*x5, 'ObjectiveSense', 'min');

prob.Constraints.c1 = 23*x1 + 171*x2 + 65*x3 + 112*x4 + 188*x5 >= 2000;
prob.Constraints.c2 = 0.1*x1 + 0.2*x2 + 0*x3 + 9.3*x4 + 16*x5 >= 50;
prob.Constraints.c3 = 0.6*x1 + 3.7*x2 + 2.2*x3 + 7*x4 + 7.7*x5 >= 100;
prob.Constraints.c4 = 6*x1 + 30*x2 + 13*x3 + 0*x4 + 2*x5 >= 250;

problem = prob2struct(prob);

[x, fval, exitflag, output] = linprog(problem);
funcsol = 0.14*x(1) + 0.12*x(2) + 0.2*x(3) + 0.75*x(4) + 0.15*x(5);

x, funcsol
