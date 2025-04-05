import numpy as np
import matplotlib.pyplot as plt
import functions as fn
from matplotlib.animation import FuncAnimation
from sympy import limit
import cupy as cp


def limit_check(fx, t, y, a, F, f, phi, mu, tau):
    x_right_vals = np.linspace(0.01, 0.00001, 10)
    x_left_vals = np.linspace(-0.01, -0.00001, 10)
    limits_right = [fx(t, x_small, y, a, F, f, phi, mu, tau) for x_small in x_right_vals]
    limits_left = [fx(t, x_small, y, a, F, f, phi, mu, tau) for x_small in x_left_vals]

    # Define a tolerance for convergence
    tolerance = 1e-1

    # Check for convergence by comparing differences between consecutive values
    differences_right = np.abs(np.diff(limits_right))
    differences_left = np.abs(np.diff(limits_left))
    converged_right = np.all(differences_right < tolerance)
    converged_left = np.all(differences_left < tolerance)

    return converged_left and converged_right


# --------------------------------------------------------

# def limit_check_gpu(fx, t, y):
#     x_right_vals = cp.linspace(0.01, 0.00001, 100)  # Use CuPy for GPU arrays
#     x_left_vals = cp.linspace(-0.01, -0.00001, 100)  # Use CuPy for GPU arrays

#     limits_right = cp.array([fx(t, x_small, y) for x_small in x_right_vals])  # Use CuPy arrays
#     limits_left = cp.array([fx(t, x_small, y) for x_small in x_left_vals])  # Use CuPy arrays

#     # Define a tolerance for convergence
#     tolerance = 1e-5

#     # Check for convergence by comparing differences between consecutive values
#     differences_right = cp.abs(cp.diff(limits_right))  # Use CuPy's diff and abs
#     differences_left = cp.abs(cp.diff(limits_left))  # Use CuPy's diff and abs

#     converged_right = cp.all(differences_right < tolerance)  # Use CuPy's all function
#     converged_left = cp.all(differences_left < tolerance)  # Use CuPy's all function

#     return converged_left and converged_right
