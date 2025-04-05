import functions as fn
from sympy import symbols, Eq, solve


# Compute mu0 (value of behavior change for roots t0)
def compute_mu0(mu, a, b, tau):
    mu0 = (4*a*b*tau)/(b**2 * tau**2 + 4*a**2) if mu >= 0 else -(4*a*b*tau)/(b**2 * tau**2 + 4*a**2)
    return mu0


# Compute t_critical (roots t0 have to be <= than t critical)
def compute_tcr(mu, a, b, tau):
    if mu >= 0:
        tcr = [(1/mu)*(tau/2 - a/b) + tau/2 + a/b]
    else:
        tcr = [(1/mu)*(tau/2 - a/b) + tau/2 + a/b, (1/mu)*(tau/2 + a/b) + tau/2 - a/b]
    return tcr


# Compute roots t0 (time when f(t) changes its sign)
def compute_roots_t0(mu, a, b, tau):
    t0 = symbols('t0')
    equation = Eq(b*mu*t0**2 - (b*mu*tau + 2*a*mu)*t0 + (a + a*mu)*tau, 0)
    solutions = solve(equation, t0)
    return solutions


# Compute y0 (value where we intercept y-axis)
def compute_y0(t0, mu, b, tau):
    y0 = -b*mu*t0 + (b + b*mu)*tau/2
    return y0
