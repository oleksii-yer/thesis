import numpy as np
import matplotlib.pyplot as plt
import functions as fn
from matplotlib.animation import FuncAnimation
from utility import limit_check, limit_check_gpu
from numba import jit, cuda


def euler(fx, fy, t0, x0, y0, h, t_end):
    t_values = [t0]
    x_values = [x0]
    y_values = [y0]

    t = t0
    x = x0
    y = y0

    while t <= t_end:
        x_new = x + h * fx(t, x, y)
        y_new = y + h * fy(t, x, y)

        t += h

        t_values.append(t)
        x_values.append(x_new)
        y_values.append(y_new)

        x = x_new
        y = y_new
    
    return np.array(t_values), np.array(x_values), np.array(y_values)


def euler_slide(fx, fy, t0, x0, y0, h, t_end):
    t_values = [t0]
    x_values = [x0]
    y_values = [y0]
    slide_zone = []

    t = t0
    x = x0
    y = y0

    while t <= t_end:
        if (x == 0) and (fx(t, 10e-5, y) < 0) and (fx(t, -10e-5, y) > 0):
            if limit_check(fx, t, y):
                x_new = 0
                y_new = (fy(t, 10e-5, y) + fy(t, -10e-5, y))/2
            else:
                raise(KeyError)
        else:        
            x_new = x + h * fx(t, x, y)
            y_new = y + h * fy(t, x, y)

        if x*x_new < 0:
            h_tilda = -x/fx(t, x, y)
            x_new = 0
            y_new = y + h_tilda*fy(t, x, y)
            t -= h - h_tilda

        slide_zone.append(((fx(t, -10e-5, y)-y)*(-1), (fx(t, 10e-5, y)-y)*(-1)))

        t += h

        t_values.append(t)
        x_values.append(x_new)
        y_values.append(y_new)

        x = x_new
        y = y_new
    
    slide_zone.append(0)
    return np.array(t_values), np.array(x_values), np.array(y_values), slide_zone


def runge_kutta(fx, fy, t0, x0, y0, h, t_end):
    t_values = [t0]
    x_values = [x0]
    y_values = [y0]

    t = t0
    x = x0
    y = y0

    while t <= t_end:
        k1_x = fx(t, x, y)
        k2_x = fx(t + h/2, x + h/2 * k1_x, y)
        k3_x = fx(t + h/2, x + h/2 * k2_x, y)
        k4_x = fx(t + h, x + h * k3_x, y)

        k1_y = fy(t, x, y)
        k2_y = fy(t + h/2, x, y + h/2 * k1_y)
        k3_y = fy(t + h/2, x, y + h/2 * k2_y)
        k4_y = fy(t + h, x, y + h * k3_y)

        x_new = x + h/6 * (k1_x + 2*k2_x + 2*k3_x + k4_x)
        y_new = y + h/6 * (k1_y + 2*k2_y + 2*k3_y + k4_y)

        t += h

        t_values.append(t)
        x_values.append(x_new)
        y_values.append(y_new)

        x = x_new
        y = y_new
    
    return np.array(t_values), np.array(x_values), np.array(y_values)

import cupy as cp


def runge_kutta_slide(fx, fy, t0, x0, y0, h, t_end, F_type, f_type, a, b, phi, mu, tau, t0_shift=False, period=None):
    # At lest keeps errors consistent
    if t0_shift:
        t0 = t0 % period

    t_values = [t0]
    x_values = [x0]
    y_values = [y0]
    slide_zone = []

    F, f  = None, None

    if f_type == 'meand':
        f = fn.f_meand
    elif f_type == 'sin':
        f = fn.sin

    if F_type == 'collapsed':
        F = fn.F_collapsed
    elif F_type == 'open':
        F = fn.F_open

    t = t0
    x = x0
    y = y0

    while t < t_end:
        if (x == 0) and (fx(t, 10e-5, y, a, F, f, phi, mu, tau) < 0) and (fx(t, -10e-5, y, a, F, f, phi, mu, tau) > 0):
            if limit_check(fx, t, y, a, F, f, phi, mu, tau):
                x_new = 0
                y_new = (fy(t, 10e-5, y, b, F, f, phi, mu, tau) + fy(t, -10e-5, y, b, F, f, phi, mu, tau))/2
            else:
                raise(KeyError)
        else:        
            k1_x = fx(t, x, y, a, F, f, phi, mu, tau)
            k2_x = fx(t + h/2, x + h/2 * k1_x, y, a, F, f, phi, mu, tau)
            k3_x = fx(t + h/2, x + h/2 * k2_x, y, a, F, f, phi, mu, tau)
            k4_x = fx(t + h, x + h * k3_x, y, a, F, f, phi, mu, tau)

            k1_y = fy(t, x, y, b, F, f, phi, mu, tau)
            k2_y = fy(t + h/2, x, y + h/2 * k1_y, b, F, f, phi, mu, tau)
            k3_y = fy(t + h/2, x, y + h/2 * k2_y, b, F, f, phi, mu, tau)
            k4_y = fy(t + h, x, y + h * k3_y, b, F, f, phi, mu, tau)

            x_new = x + h/6 * (k1_x + 2*k2_x + 2*k3_x + k4_x)
            y_new = y + h/6 * (k1_y + 2*k2_y + 2*k3_y + k4_y)

        if x*x_new < 0:
            h_tilda = -x/fx(t, x, y, a, F, f, phi, mu, tau)
            x_new = 0
            k1_y = fy(t, x, y, b, F, f, phi, mu, tau)
            k2_y = fy(t + h/2, x, y + h/2 * k1_y, b, F, f, phi, mu, tau)
            k3_y = fy(t + h/2, x, y + h/2 * k2_y, b, F, f, phi, mu, tau)
            k4_y = fy(t + h, x, y + h * k3_y, b, F, f, phi, mu, tau)

            y_new = y + h/6 * (k1_y + 2*k2_y + 2*k3_y + k4_y)
            t -= h - h_tilda

        slide_zone.append(((fx(t, -10e-5, y, a, F, f, phi, mu, tau)-y)*(-1), (fx(t, 10e-5, y, a, F, f, phi, mu, tau)-y)*(-1)))

        t += h

        t_values.append(t)
        x_values.append(x_new)
        y_values.append(y_new)

        x = x_new
        y = y_new

    slide_zone.append(0)
    return np.array(t_values), np.array(x_values), np.array(y_values), slide_zone


def runge_kutta_iter(fx, fy, t0, x0, y0, h, t_end, F_type, f_type, a, b, phi, mu, tau, t0_shift=False, period=None):
    if t0_shift:
        t0 = t0 % period

    # t_values = [t0]
    # x_values = [x0]
    # y_values = [y0]

    F, f  = None, None

    if f_type == 'meand':
        f = fn.f_meand
    elif f_type == 'sin':
        f = fn.sin

    if F_type == 'collapsed':
        F = fn.F_collapsed
    elif F_type == 'open':
        F = fn.F_open

    t = t0
    x = x0
    y = y0

    print(x, y)
    while True:
        if (x == 0) and (fx(t, 10e-5, y, a, F, f, phi, mu, tau) < 0) and (fx(t, -10e-5, y, a, F, f, phi, mu, tau) > 0):
            if limit_check(fx, t, y, a, F, f, phi, mu, tau):
                x_new = 0
                y_new = (fy(t, 10e-5, y, b, F, f, phi, mu, tau) + fy(t, -10e-5, y, b, F, f, phi, mu, tau))/2
            else:
                raise(KeyError)
        else:        
            k1_x = fx(t, x, y, a, F, f, phi, mu, tau)
            k2_x = fx(t + h/2, x + h/2 * k1_x, y, a, F, f, phi, mu, tau)
            k3_x = fx(t + h/2, x + h/2 * k2_x, y, a, F, f, phi, mu, tau)
            k4_x = fx(t + h, x + h * k3_x, y, a, F, f, phi, mu, tau)

            k1_y = fy(t, x, y, b, F, f, phi, mu, tau)
            k2_y = fy(t + h/2, x, y + h/2 * k1_y, b, F, f, phi, mu, tau)
            k3_y = fy(t + h/2, x, y + h/2 * k2_y, b, F, f, phi, mu, tau)
            k4_y = fy(t + h, x, y + h * k3_y, b, F, f, phi, mu, tau)

            x_new = x + h/6 * (k1_x + 2*k2_x + 2*k3_x + k4_x)
            y_new = y + h/6 * (k1_y + 2*k2_y + 2*k3_y + k4_y)

        if x*x_new < 0:
            h_tilda = -x/fx(t, x, y, a, F, f, phi, mu, tau)
            x_new = 0
            k1_y = fy(t, x, y, b, F, f, phi, mu, tau)
            k2_y = fy(t + h/2, x, y + h/2 * k1_y, b, F, f, phi, mu, tau)
            k3_y = fy(t + h/2, x, y + h/2 * k2_y, b, F, f, phi, mu, tau)
            k4_y = fy(t + h, x, y + h * k3_y, b, F, f, phi, mu, tau)

            y_new = y + h/6 * (k1_y + 2*k2_y + 2*k3_y + k4_y)
            t -= h - h_tilda

        t += h

        # t_values.append(t)
        # x_values.append(x_new)
        # y_values.append(y_new)

        x = x_new
        y = y_new

        if (x != 0 and y < 0):
            break

    return x, y

# --------------------------------------------------------------

# @cuda.jit
# def runge_kutta_kernel(t0, x0, y0, h, t_end, a, b, phi, mu, tau, x_out, y_out):
#     i = cuda.grid(1)
#     if i >= t0.shape[0]:
#         return
#     t = t0[i]
#     x = x0[i]
#     y = y0[i]
#     idx = 0
    
#     while t <= t_end:
#         k1_x = dx_dt(t, x, y, a, F_collapsed, f_meand, phi, mu, tau)
#         k2_x = dx_dt(t + h / 2, x + h / 2 * k1_x, y, a, F_collapsed, f_meand, phi, mu, tau)
#         k3_x = dx_dt(t + h / 2, x + h / 2 * k2_x, y, a, F_collapsed, phi, mu, tau)
#         k4_x = dx_dt(t + h, x + h * k3_x, y, a, F_collapsed, phi, mu, tau)
        
#         k1_y = dy_dt(t, x, y, b, F_collapsed, f_meand, phi, mu, tau)
#         k2_y = dy_dt(t + h / 2, x, y + h / 2 * k1_y, b, F_collapsed, f_meand, phi, mu, tau)
#         k3_y = dy_dt(t + h / 2, x, y + h / 2 * k2_y, b, F_collapsed, f_meand, phi, mu, tau)
#         k4_y = dy_dt(t + h, x, y + h * k3_y, b, F_collapsed, f_meand, phi, mu, tau)
        
#         x_new = x + h / 6 * (k1_x + 2 * k2_x + 2 * k3_x + k4_x)
#         y_new = y + h / 6 * (k1_y + 2 * k2_y + 2 * k3_y + k4_y)
        
#         x_out[i, idx] = x_new
#         y_out[i, idx] = y_new
#         idx += 1
        
#         t += h
#         x = x_new
#         y = y_new