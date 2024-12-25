import numpy as np
import matplotlib.pyplot as plt
import functions as fn
from matplotlib.animation import FuncAnimation
from utility import limit_check
import ipywidgets as widgets
from ipywidgets import interact
import sympy


def euler(fx, fy, t_0, x_0, y_0, h, t_end):
    t_values = [t_0]
    x_values = [x_0]
    y_values = [y_0]

    t = t_0
    x = x_0
    y = y_0

    while t <= t_end:
        x_new = x + h*fx(t, x, y)
        y_new = y + h*fy(t, x, y)

        t += h

        t_values.append(t)
        x_values.append(x_new)
        y_values.append(y_new)

        x = x_new
        y = y_new
    
    return np.array(t_values), np.array(x_values), np.array(y_values)


def euler_slide(fx, fy, t_0, x_0, y_0, h, t_end):
    t_values = [t_0]
    x_values = [x_0]
    y_values = [y_0]
    slide_zone = []

    t = t_0
    x = x_0
    y = y_0

    while t <= t_end:
        if (x == 0) and (fx(t, 10e-5, y) < 0) and (fx(t, -10e-5, y) > 0):
                if limit_check(fx, t, y):
                    x_new = 0
                    y_new = (fy(t, 10e-5, y) + fy(t, -10e-5, y))/2
                else:
                    raise(KeyError)
        else:        
            x_new = x + h*fx(t, x, y)
            y_new = y + h*fy(t, x, y)

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