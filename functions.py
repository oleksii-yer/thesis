import numpy as np
from constants import params

def F_collapsed(x):
    if x < 0:
        return -1
    if x > 0:
        return 1
    else:
        return 0


def F(x):
    if x < -1:
        return -1
    if x > 1:
        return 1
    else:
        return x
    

def f_meand(t, phi=params['phi'], mu=params['mu'], tau=params['tau']):
    if phi >= 0:
        if 0 <= t % (2*tau) < phi:
            return -mu
        if phi <= t % (2*tau) < phi + tau:
            #print('+')
            return mu
        if phi + tau <= t % (2*tau) <= 2*tau:
            #print('-')
            return -mu
    else:
        if 0 <= t % (2*tau) < tau + phi:
            return mu
        if tau + phi <= t % (2*tau) < 2*tau + phi:
            return -mu
        if 2*tau + phi <= t % (2*tau) <= 2*tau:
            return mu
        

def sin_mu(t, mu=params['mu']):
    return mu*np.sin(t)


def dx_dt(t, x, y, a=params['a'], F = F_collapsed, f = f_meand):
    return y - a*(F(x) + f(t))


def dy_dt(t, x, y, b=params['b'], F = F_collapsed, f = f_meand):
    return -b*(F(x) + f(t))


def dx_dt_reverse(t, x, y, a=params['a'], F = F_collapsed, f = f_meand):
    return -y + a*(F(x) + f(-t))


def dy_dt_reverse(t, x, y, b=params['b'], F = F_collapsed, f = f_meand):
    return b*(F(x) + f(-t))
