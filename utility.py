import numpy as np
import matplotlib.pyplot as plt
import functions as fn
from matplotlib.animation import FuncAnimation
from sympy import limit


def limit_check(fx, t, y):
    x_right_vals = np.linspace(0.01, 0.00001, 100)
    x_left_vals = np.linspace(-0.01, -0.00001, 100)
    limits_right = [fx(t, x_small, y) for x_small in x_right_vals]
    limits_left = [fx(t, x_small, y) for x_small in x_left_vals]

    # Define a tolerance for convergence
    tolerance = 1e-6

    # Check for convergence by comparing differences between consecutive values
    differences_right = np.abs(np.diff(limits_right))
    differences_left = np.abs(np.diff(limits_left))
    converged_right = np.all(differences_right < tolerance)
    converged_left = np.all(differences_left < tolerance)

    return converged_left and converged_right