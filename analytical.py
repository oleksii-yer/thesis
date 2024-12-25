import functions as fn
from constants import params
from sympy import symbols, Eq, solve


# Compute mu0 (value of behavior change for roots t0)
def compute_mu0(a=params['a'], b=params['b'], tau=params['tau']):
    mu0 = (4*a*b*tau)/(b**2 * tau**2 + 4*a**2) if params['mu'] >= 0 else -(4*a*b*tau)/(b**2 * tau**2 + 4*a**2)
    return mu0


# Compute t_critical (roots t0 have to be <= than t critical)
def compute_tcr(mu=params['mu'], a=params['a'], b=params['b'], tau=params['tau']):
    tcr = (1/mu)*(tau/2 - a/b) + tau/2 + a/b
    return tcr


# Compute roots t0 (time when f(t) changes its sign)
def compute_roots_t0(mu=params['mu'], a=params['a'], b=params['b'], tau=params['tau']):
    t0 = symbols('t0')
    equation = Eq(b*mu*t0**2 - (b*mu*tau + 2*a*mu)*t0 + (a + a*mu)*tau, 0)
    solutions = solve(equation, t0)
    return solutions


# Compute y0 (value where we intercept y-axis)
def compute_y0(t0, mu=params['mu'], b=params['b'], tau=params['tau']):
    y0 = -b*mu*t0 + (b + b*mu)*tau/2
    return y0