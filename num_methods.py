import numpy as np
import functions as fn
from utility import limit_check
import analytical as an


class NoConverge(Exception):
    """Custom exception for something specific."""
    pass


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


def runge_kutta_slide(fx, fy, t0, x0, y0, h, t_end, F_type, f_type, a, b, mu, tau):
    t_values = [t0]
    x_values = [x0]
    y_values = [y0]

    F, f  = None, None

    if f_type == 'meand':
        f = fn.f_meand
    elif f_type == 'sin':
        f = fn.sin
    else:
        raise NameError

    if F_type == 'closed':
        F = fn.F_closed
    elif F_type == 'open':
        F = fn.F_open
    elif F_type == 'tanh':
        F = np.tanh
    else:
        raise NameError

    t = t0
    x = x0
    y = y0


    while t < t_end:
        if F_type == 'closed':
            if (x == 0) and (fx(t, 10e-5, y, a, F, f, mu, tau) < 0) and (fx(t, -10e-5, y, a, F, f, mu, tau) > 0):
                x_new = 0
                k1_y = fy(t, -10e-5, y, b, F, f, mu, tau)
                k2_y = fy(t + h/2, -10e-5, y + h/2 * k1_y, b, F, f, mu, tau)
                k3_y = fy(t + h/2, -10e-5, y + h/2 * k2_y, b, F, f, mu, tau)
                k4_y = fy(t + h, -10e-5, y + h * k3_y, b, F, f, mu, tau)

                y_incr_left = h/6 * (k1_y + 2*k2_y + 2*k3_y + k4_y)

                k1_y = fy(t, 10e-5, y, b, F, f, mu, tau)
                k2_y = fy(t + h/2, 10e-5, y + h/2 * k1_y, b, F, f, mu, tau)
                k3_y = fy(t + h/2, 10e-5, y + h/2 * k2_y, b, F, f, mu, tau)
                k4_y = fy(t + h, 10e-5, y + h * k3_y, b, F, f, mu, tau)

                y_incr_right = h/6 * (k1_y + 2*k2_y + 2*k3_y + k4_y)
                if ((y_incr_left + y_incr_right) / 2) < an.compute_slide_botedge(t, a, mu, tau, f_type):
                    y_new = an.compute_slide_botedge(t, a, mu, tau, f_type) - 10e-5
                    h_coef = np.abs((y_new - y)/(((y_incr_left + y_incr_right) / 2) - y))
                    h_tilda = h*h_coef
                    t -= h - h_tilda
                elif ((y_incr_left + y_incr_right) / 2) > an.compute_slide_topedge(t, a, mu, tau, f_type):
                    y_new = an.compute_slide_topedge(t, a, mu, tau, f_type) + 10e-5
                    h_coef = np.abs((y_new - y)/(((y_incr_left + y_incr_right) / 2) - y))
                    h_tilda = h*h_coef
                    t -= h - h_tilda
                else:
                    y_new = y + (y_incr_left + y_incr_right) / 2
            else:        
                k1_x = fx(t, x, y, a, F, f, mu, tau)
                k2_x = fx(t + h/2, x + h/2 * k1_x, y, a, F, f, mu, tau)
                k3_x = fx(t + h/2, x + h/2 * k2_x, y, a, F, f, mu, tau)
                k4_x = fx(t + h, x + h * k3_x, y, a, F, f, mu, tau)

                k1_y = fy(t, x, y, b, F, f, mu, tau)
                k2_y = fy(t + h/2, x, y + h/2 * k1_y, b, F, f, mu, tau)
                k3_y = fy(t + h/2, x, y + h/2 * k2_y, b, F, f, mu, tau)
                k4_y = fy(t + h, x, y + h * k3_y, b, F, f, mu, tau)

                x_new = x + h/6 * (k1_x + 2*k2_x + 2*k3_x + k4_x)
                y_new = y + h/6 * (k1_y + 2*k2_y + 2*k3_y + k4_y)
            
        else:        
            k1_x = fx(t, x, y, a, F, f, mu, tau)
            k2_x = fx(t + h/2, x + h/2 * k1_x, y, a, F, f, mu, tau)
            k3_x = fx(t + h/2, x + h/2 * k2_x, y, a, F, f, mu, tau)
            k4_x = fx(t + h, x + h * k3_x, y, a, F, f, mu, tau)

            k1_y = fy(t, x, y, b, F, f, mu, tau)
            k2_y = fy(t + h/2, x, y + h/2 * k1_y, b, F, f, mu, tau)
            k3_y = fy(t + h/2, x, y + h/2 * k2_y, b, F, f, mu, tau)
            k4_y = fy(t + h, x, y + h * k3_y, b, F, f, mu, tau)

            x_new = x + h/6 * (k1_x + 2*k2_x + 2*k3_x + k4_x)
            y_new = y + h/6 * (k1_y + 2*k2_y + 2*k3_y + k4_y)

        if x*x_new < 0:
            slope =(y_new - y)/(x_new - x)
            y_tilda = slope*(-x)+y
            h_coef = np.sqrt((0-x)**2+(y_tilda-y)**2)/np.sqrt((x_new-x)**2+(y_new-y)**2)
            h_tilda = h*h_coef
            x_new = 0

            y_new = y_tilda
            print(y_new)

            t -= h - h_tilda


        t += h

        t_values.append(t)
        x_values.append(x_new)
        y_values.append(y_new)

        x = x_new
        y = y_new
    
    
    return np.array(t_values), np.array(x_values), np.array(y_values)


def runge_kutta_iter(fx, fy, t0, x0, y0, h, F_type, f_type, a, b, mu, tau):
    F, f  = None, None

    if f_type == 'meand':
        f = fn.f_meand
    elif f_type == 'sin':
        f = fn.sin

    if F_type == 'closed':
        F = fn.F_closed
    elif F_type == 'open':
        F = fn.F_open

    t = t0
    x = x0
    y = y0

    while t <= t0 + 2*tau:
        if (x == 0) and (fx(t, 10e-5, y, a, F, f, mu, tau) < 0) and (fx(t, -10e-5, y, a, F, f, mu, tau) > 0):
            x_new = 0
            k1_y = fy(t, -10e-5, y, b, F, f, mu, tau)
            k2_y = fy(t + h/2, -10e-5, y + h/2 * k1_y, b, F, f, mu, tau)
            k3_y = fy(t + h/2, -10e-5, y + h/2 * k2_y, b, F, f, mu, tau)
            k4_y = fy(t + h, -10e-5, y + h * k3_y, b, F, f, mu, tau)

            y_incr_left = h/6 * (k1_y + 2*k2_y + 2*k3_y + k4_y)

            k1_y = fy(t, 10e-5, y, b, F, f, mu, tau)
            k2_y = fy(t + h/2, 10e-5, y + h/2 * k1_y, b, F, f, mu, tau)
            k3_y = fy(t + h/2, 10e-5, y + h/2 * k2_y, b, F, f, mu, tau)
            k4_y = fy(t + h, 10e-5, y + h * k3_y, b, F, f, mu, tau)

            y_incr_right = h/6 * (k1_y + 2*k2_y + 2*k3_y + k4_y)

            if ((y_incr_left + y_incr_right) / 2) < an.compute_slide_botedge(t, a, mu, tau, f_type):
                y_new = an.compute_slide_botedge(t, a, mu, tau, f_type) - 10e-5
                h_coef = np.abs((y_new - y)/(((y_incr_left + y_incr_right) / 2) - y))
                h_tilda = h*h_coef
                t -= h - h_tilda
            elif ((y_incr_left + y_incr_right) / 2) > an.compute_slide_topedge(t, a, mu, tau, f_type):
                y_new = an.compute_slide_topedge(t, a, mu, tau, f_type) + 10e-5
                h_coef = np.abs((y_new - y)/(((y_incr_left + y_incr_right) / 2) - y))
                h_tilda = h*h_coef
                t -= h - h_tilda
            else:
                y_new = y + (y_incr_left + y_incr_right) / 2
        else:
            k1_x = fx(t, x, y, a, F, f, mu, tau)
            k2_x = fx(t + h/2, x + h/2 * k1_x, y, a, F, f, mu, tau)
            k3_x = fx(t + h/2, x + h/2 * k2_x, y, a, F, f, mu, tau)
            k4_x = fx(t + h, x + h * k3_x, y, a, F, f, mu, tau)

            k1_y = fy(t, x, y, b, F, f, mu, tau)
            k2_y = fy(t + h/2, x, y + h/2 * k1_y, b, F, f, mu, tau)
            k3_y = fy(t + h/2, x, y + h/2 * k2_y, b, F, f, mu, tau)
            k4_y = fy(t + h, x, y + h * k3_y, b, F, f, mu, tau)

            x_new = x + h/6 * (k1_x + 2*k2_x + 2*k3_x + k4_x)
            y_new = y + h/6 * (k1_y + 2*k2_y + 2*k3_y + k4_y)

        if x*x_new < 0:
            slope =(y_new - y)/(x_new - x)
            y_tilda = slope*(-x)+y
            h_coef = np.sqrt((0-x)**2+(y_tilda-y)**2)/np.sqrt((x_new-x)**2+(y_new-y)**2)
            h_tilda = h*h_coef
            x_new = 0

            y_new = y_tilda

            t -= h - h_tilda

        t += h
        x = x_new
        y = y_new


    return x, y
