import numpy as np
import matplotlib.pyplot as plt
import functions as fn
from num_methods import euler, euler_slide
from ploting import plot_anim, plot_basic
import analytical as an
from constants import params


if __name__ == '__main__':
    t0 = 4.87885582211888  # Initial condition for t
    x0 = 0  # Initial condition for x
    y0 = -4  # Initial condition for y
    h = 0.01  # Time step
    t_end = 200  # End time

    # Solve using Euler's method
    t_values, x_values, y_values, slide_values = euler_slide(fn.dx_dt, fn.dy_dt, t0, x0, y0, h, t_end)

    plot_anim(t_values, x_values, y_values, slide_values, include_slide=True)

    # t_values, x_values, y_values = euler(fn.dx_dt, fn.dy_dt, t0, x0, y0, h, t_end)

    # plot_basic(t_values, x_values, y_values)

    