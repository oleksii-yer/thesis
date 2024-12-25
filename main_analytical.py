import analytical as an
from constants import params


if __name__ == '__main__':
    print('mu0 =', an.compute_mu0())
    print('tcr =', an.compute_tcr())
    print('t0 solutions =', an.compute_roots_t0())
    print('y0 =', [an.compute_y0(t) for t in an.compute_roots_t0()])

    print('\n')
    print('t0 for i.c.:', [params['tau'] - t for t in an.compute_roots_t0()])
    print('y0 fro i.c.:', [-an.compute_y0(t) for t in an.compute_roots_t0()])