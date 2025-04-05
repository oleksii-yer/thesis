import numpy as np
from numba import cuda

 
def F_collapsed(x):
    if x < 0:
        return -1
    if x > 0:
        return 1
    else:
        return 0
    
    
def F_open(x):
    if x < -1:
        return -1
    if x > 1:
        return 1
    else:
        return x

  
def f_meand(t, phi, mu, tau):
    if phi >= 0:
        if 0 <= t % (2*tau) < phi:
            return -mu
        if phi <= t % (2*tau) < phi + tau:
            return mu
        if phi + tau <= t % (2*tau) <= 2*tau:
            return -mu
    else:
        if 0 <= t % (2*tau) < tau + phi:
            return mu
        if tau + phi <= t % (2*tau) < 2*tau + phi:
            return -mu
        if 2*tau + phi <= t % (2*tau) <= 2*tau:
            return mu
        

def sin(t, phi, mu, tau):
    return mu*np.sin(t / tau) + phi


def dx_dt(t, x, y, a, F, f, phi, mu, tau):
    return y - a*(F(x) + f(t, phi, mu, tau))


def dy_dt(t, x, y, b, F, f, phi, mu, tau):
    return -b*(F(x) + f(t, phi, mu, tau))


def dx_dt_reverse(t, x, y, a, F, f):
    return -y + a*(F(x) + f(-t))


def dy_dt_reverse(t, x, y, b, F, f):
    return b*(F(x) + f(-t))


# ------------------------------------------------

# @cuda.jit(device=True)
# def dx_dt_gpu(t, x, y, a, F, f, phi, mu, tau):
#     return y - a*(F(x) + f(t, phi, mu, tau))


# @cuda.jit(device=True)
# def dy_dt_gpu(t, x, y, b, F, f, phi, mu, tau):
#     return -b*(F(x) + f(t, phi, mu, tau))


# @cuda.jit(device=True) 
# def F_collapsed_gpu(x):
#     if x < 0:
#         return -1
#     if x > 0:
#         return 1
#     else:
#         return 0
    
    
# @cuda.jit(device=True) 
# def F_open_gpu(x):
#     if x < -1:
#         return -1
#     if x > 1:
#         return 1
#     else:
#         return x


# @cuda.jit(device=True)    
# def f_meand_gpu(t, phi, mu, tau):
#     if phi >= 0:
#         if 0 <= t % (2*tau) < phi:
#             return -mu
#         if phi <= t % (2*tau) < phi + tau:
#             return mu
#         if phi + tau <= t % (2*tau) <= 2*tau:
#             return -mu
#     else:
#         if 0 <= t % (2*tau) < tau + phi:
#             return mu
#         if tau + phi <= t % (2*tau) < 2*tau + phi:
#             return -mu
#         if 2*tau + phi <= t % (2*tau) <= 2*tau:
#             return mu