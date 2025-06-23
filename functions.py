import numpy as np

 
def F_closed(x):
    if x < 0:
        return -1
    if x > 0:
        return 1
    else:
        return 0
    
    
def F_open(x):
    if x < -2:
        return -1
    if x > 2:
        return 1
    else:
        return x/2


def f_meand(t, mu, tau):
    if 0 <= t % (2*tau) < tau:
        return mu
    if tau <= t % (2*tau) <= 2*tau:
        return -mu
        

def sin(t, mu, tau):
    return mu*np.sin(np.pi*t / tau)


def dx_dt(t, x, y, a, F, f, mu, tau):
    return y - a*(F(x) + f(t, mu, tau))


def dy_dt(t, x, y, b, F, f, mu, tau):
    return -b*(F(x) + f(t, mu, tau))


def dx_dt_reverse(t, x, y, a, F, f):
    return -y + a*(F(x) + f(-t))


def dy_dt_reverse(t, x, y, b, F, f):
    return b*(F(x) + f(-t))
