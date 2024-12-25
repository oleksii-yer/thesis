from sympy import symbols, solveset, S, Interval

# Define the symbol and function
x = symbols('x')
f = -x**2 - 1  # Example function

# Define the interval
interval = Interval(-2, -1)

# Solve for where the function is non-negative
non_negative_set = solveset(f >= 0, x, domain=interval)

# Check if the function is negative on the interval
is_negative = non_negative_set.is_EmptySet

print(f"The function is negative on the interval {interval}: {is_negative}")
