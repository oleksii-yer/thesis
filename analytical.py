import functions as fn
import numpy as np


# Compute mu0 (roots t0 born)
def compute_mu0(a, b, tau, f_type):
    if f_type == 'meand':
        mu0 = (4*a*b*tau)/(b**2 * tau**2 + 4*a**2)
    elif f_type == 'sin':
        mu0 = (a*np.pi/b)/(2*np.sqrt((tau/np.pi)**2 + (a/b)**2))
        
    return mu0
    
# Compute mu1 (in case of sin value when one root remains)
def compute_mu1(a, b, tau):
    mu1 = compute_mu0(a, b, tau, 'sin') * np.sqrt(1 + ((a/b - tau/2)/(a*np.pi/(2*b)))**2)
    return mu1

# Compute t_critical (roots t0 have to be > than t critical)
def compute_tcr(mu, a, b, tau, f_type):
    if f_type == 'meand':
        tcr = (1/mu)*(a/b - tau/2) + tau/2 - a/b

    elif f_type == 'sin':
        R = mu*np.sqrt((tau/np.pi)**2 + (a/b)**2)
        C = a/b - tau/2
        if np.abs(C/R) < 1:
            alpha = np.arctan((b*tau)/(a*np.pi))
            tcr1 = (np.arcsin(C/R) + alpha) * (tau/np.pi)
            tcr2 = (np.pi - np.arcsin(C/R) + alpha) * (tau/np.pi)
            tcr = [tcr1, tcr2]
        else:
            tcr = []
    return tcr


# Compute roots t0 (time when f(t) changes its sign)
def compute_roots_t0(mu, a, b, tau, f_type, tcr_filter=True):
    if f_type == 'meand':
        discrim = (2*a*mu - b*mu*tau)**2 - 4*b*mu*(a - a*mu)*tau
        root1 = ((b*mu*tau - 2*a*mu) - np.sqrt(discrim)) / (2*b*mu)
        root2 = ((b*mu*tau - 2*a*mu) + np.sqrt(discrim)) / (2*b*mu)
        solutions = np.array([root1, root2], dtype=float)
        if tcr_filter:
            solutions = solutions[(solutions > compute_tcr(mu, a, b, tau, f_type))]
        return solutions
    
    elif f_type == 'sin':
        R = np.sqrt((b*mu*tau**2)**2 + (a*mu*tau*np.pi)**2)
        C = (a*tau*np.pi**2)/2
        if np.abs(C/R) <= 1 + 1e-8:
            alpha = np.arctan((a*np.pi)/(b*tau))
            root1 = (np.arcsin(compute_mu0(a, b, tau, 'sin')/mu) - alpha) * (tau/np.pi)
            root2 = (np.pi - np.arcsin(compute_mu0(a, b, tau, 'sin')/mu) - alpha) * (tau/np.pi)
            solutions = np.array([root1, root2], dtype=float)
            if tcr_filter:
                if compute_tcr(mu, a, b, tau, f_type) != []:
                    solutions = solutions[(solutions > compute_tcr(mu, a, b, tau, f_type)[0]) & (solutions < compute_tcr(mu, a, b, tau, f_type)[1])]
                else:
                    pass
        else:
            solutions = []
        return solutions

# Compute y0 (y-intercept of attracting cycle)
def compute_y0(t0, mu, b, tau, f_type):
    if f_type == 'meand':
        y0 = b*mu*t0 + (b - b*mu)*tau/2
    elif f_type == 'sin':
        y0 = (b*tau)/2 - np.cos(np.pi*t0/tau)*(b*mu*tau)/np.pi

    return y0


def compute_slide_window(a, mu):
    dist = a*(mu - 1) - a*(-mu + 1)
    return dist


def compute_slide_topedge(t, a, mu, tau, f_type):
    f  = None

    if f_type == 'meand':
        f = fn.f_meand
    elif f_type == 'sin':
        f = fn.sin
    
    point = a*(f(t, mu, tau) + 1)

    return point

def compute_slide_botedge(t, a, mu, tau, f_type):
    f  = None

    if f_type == 'meand':
        f = fn.f_meand
    elif f_type == 'sin':
        f = fn.sin
    
    point = a*(f(t, mu, tau) - 1)

    return point


def compute_slide_topedge_max(a, mu):
    point = a*(mu + 1)

    return point

def compute_slide_botedge_max(a, mu):
    point = a*(-mu - 1)

    return point
