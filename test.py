from functions import dy_dt
from sympy import symbols, limit


x = symbols('x')

print(limit(dy_dt(0.5, x, 1)))