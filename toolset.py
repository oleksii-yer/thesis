import numpy as np
import matplotlib.pyplot as plt
import functions as fn
import analytical as an
import num_methods as nm
from joblib import Parallel, delayed, parallel_backend
from collections import defaultdict


def t_end_map(mu, f_type):
    t_end = 0
    if f_type == 'meand':
        t_end = np.ceil(mu) * 1000
    elif f_type == 'sin':
        t_end = np.ceil(mu) * 2000

    return t_end


def find_first_zero_index(x_values):
    for i in range(1, len(x_values)+1):
        if x_values[-i] == 0:
            return i
    return len(x_values) - 1



def compute_y_intercept(t0, y0, mu, a, b, tau, F_type, f_type, h, t_end):
    x0 = 0
    _, x_values, y_values = nm.runge_kutta_slide(fn.dx_dt, fn.dy_dt, t0, x0, y0, h, t_end,
                                                    F_type, f_type, a, b, mu, tau)
    y_index = find_first_zero_index(x_values)

    if y_values[-y_index] >= 0:
        y_intercept = y_values[-y_index]
    else:
        y_index_new = find_first_zero_index(x_values[:-y_index])
        y_intercept = y_values[:-y_index][-y_index_new]

    return y_intercept



def create_diagram_point(t0_list, y0_list, mu_list, a, b, tau, F, f, h, t_end):
    num_plots = len(t0_list)
    ncols = 2 if num_plots > 1 else 1
    nrows = (num_plots + 1) // 2

    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 6))
    fig.suptitle(f"Hysteresis diagram with F - {F}, f - {f}", fontsize=16)
    if num_plots == 1:
        ax = np.array([ax])
    else:
        ax = np.ravel(ax)

    for i in range(len(t0_list)):
        with parallel_backend("loky", inner_max_num_threads=1):  # More efficient threading
            results = Parallel(n_jobs=-1, batch_size=20)( 
                delayed(compute_y_intercept)(t0_list[i], y0, mu, a, b, tau, F, f, h, t_end) 
                for mu in mu_list 
                for y0 in y0_list
            )

        mu_array = np.repeat(mu_list, len(y0_list))
        y_points = np.array(results)

        sc = ax[i].scatter(mu_array, y_points, c=np.tile(y0_list, len(mu_list)), cmap='viridis', alpha=0.5)
        fig.colorbar(sc, ax=ax[i], label=r'$y_0$ values')  # Attach colorbar to the subplot
        ax[i].plot(mu_list, np.zeros(len(mu_list)), 'k--')
        ax[i].set_title(f"t0 = {t0_list[i]}")
        ax[i].grid()

    plt.tight_layout()
    plt.show()


def create_diagram_hyst_theoretical(ax, mu_list_max, a, b, tau, f_type, alpha=0.8):
    mu_list = np.arange(0, mu_list_max+0.01, 0.01)
    mu_list = np.append(mu_list, [an.compute_mu0(a, b, tau, f_type)])
    mu_list = np.sort(mu_list)
    m0 = None
    m1 = None
    if f_type == 'meand':
        m0 = an.compute_mu0(a, b, tau, f_type)
        m1 = 1
    if f_type == 'sin':
        m0 = an.compute_mu0(a, b, tau, f_type)
        m1 = an.compute_mu1(a, b, tau)
    if (tau/2 - a/b) > 0:
        solutions_t0 = {float(mu) : an.compute_roots_t0(mu, a, b, tau, f_type) for mu in mu_list}
        solutions_y0_smaller = {float(mu) : np.min([an.compute_y0(t, mu, b, tau, f_type) for t in solutions_t0[mu]]) for mu in mu_list[(mu_list >= m0) & (mu_list < m1)]}
        solutions_y0_greater = {float(mu) : np.max([an.compute_y0(t, mu, b, tau, f_type) for t in solutions_t0[mu]]) for mu in mu_list[(mu_list > m0)]}
        mu_plot = list(solutions_y0_smaller.keys())[::-1]
        mu_plot += list(solutions_y0_greater.keys())
        y_plot = list(solutions_y0_smaller.values())[::-1] + list(solutions_y0_greater.values())
    else:
        solutions_t0 = {float(mu) : an.compute_roots_t0(mu, a, b, tau, f_type)[0] for mu in mu_list[mu_list > m1]}
        solutions_y0 = [an.compute_y0(t, mu, b, tau, f_type) for t, mu in zip(solutions_t0.values(), solutions_t0.keys())]
        mu_plot = mu_list[mu_list > m1]
        y_plot = solutions_y0

    ax.plot(mu_plot, y_plot, c='blue', alpha=alpha, linewidth=2.5, label='Theoretical')
    ax.legend()
        
    ax.set_xlabel(r'Amplitude $\mu$')
    ax.set_ylabel(r'Fixed point $y_{fixed}$')


def create_diagram_hyst_numerical(ax, t0, y0_list, mu_list, a, b, tau, F_type, f_type, h, t_end, coloring=False):
    mu_unstable_left = None
    ind_unstable_left = 0
    y_max_left = 0

    mu_unstable_right = None
    ind_unstable_right = 0
    y_max_right = 0
    if (t0 % (2*tau)) < tau:
        with parallel_backend("loky", inner_max_num_threads=1):  # More efficient threading
            results_flat = Parallel(n_jobs=-1, batch_size=10)( 
                delayed(compute_y_intercept)(t0, y0, mu, a, b, tau, F_type, f_type, h, t_end) 
                for mu in mu_list 
                for y0 in y0_list
            )
    else:
        with parallel_backend("loky", inner_max_num_threads=1):
            results_flat = Parallel(n_jobs=-1, batch_size=10)( 
                delayed(compute_y_intercept)(t0, y0, mu, a, b, tau, F_type, f_type, h, t_end) 
                for mu in mu_list 
                for y0 in -y0_list
            )

    # Group results by mu after computing
    grouped_results = defaultdict(list)
    idx = 0
    for mu in mu_list:
        for _ in y0_list:
            grouped_results[mu].append(results_flat[idx])
            idx += 1

    for j, mu in enumerate(mu_list):
        diff = np.diff(grouped_results[mu])
        upper_bound = None
        if F_type == 'closed':
            upper_bound = an.compute_slide_topedge_max(a, mu)
        elif F_type == 'open':
            upper_bound = 2*b
        elif F_type == 'tanh':
            upper_bound = 2*b
        if (np.any(diff > upper_bound) and (mu_unstable_left == None)):
            mu_unstable_left = mu
            ind_unstable_left = j
            y_max_left = np.max(grouped_results[mu_unstable_left])
        elif (np.all(np.array(grouped_results[mu], dtype=float) > upper_bound) and (mu_unstable_right == None)):
            mu_unstable_right = mu_list[j]
            ind_unstable_right = j
            y_max_right = np.min(grouped_results[mu_unstable_right])
            break

    nbh = 0.1
    unstable_mu_range = mu_list[ind_unstable_left + 1:ind_unstable_right]
    if (t0 % (2*tau)) < tau:
        unstable_y0_grouped = {mu : np.arange(0, -np.max(grouped_results[mu]), -nbh) for mu in unstable_mu_range}
    else:
        unstable_y0_grouped = {mu : np.arange(0, np.max(grouped_results[mu]), nbh) for mu in unstable_mu_range}
    unstable_y_list = []

    with parallel_backend("loky", inner_max_num_threads=1):
        unstable_results_flat = Parallel(n_jobs=-1, batch_size=10)( 
            delayed(compute_y_intercept)(np.min(an.compute_roots_t0(mu, a, b, tau, f_type, tcr_filter=False)), y0, mu, a, b, tau, F_type, f_type, h, t_end) 
            for mu in unstable_mu_range
            for y0 in unstable_y0_grouped[mu] 
        )

    grouped_unstable_results = defaultdict(list)
    idx = 0
    for mu in unstable_mu_range:
        for _ in unstable_y0_grouped[mu]:
            grouped_unstable_results[mu].append(unstable_results_flat[idx])
            idx += 1

    for mu in unstable_mu_range:
        unstable_diff = np.diff(grouped_unstable_results[mu])
        if np.any(unstable_diff > np.abs((unstable_y0_grouped[mu][-1]/2))):
            max_diff_idx = np.argmax(unstable_diff > (np.max(grouped_unstable_results[mu])) / 2)
            if (t0 % (2*tau)) < tau:
                unstable_y_list.append(-(unstable_y0_grouped[mu][max_diff_idx + 1] + unstable_y0_grouped[mu][max_diff_idx]) / 2)
            else:
                unstable_y_list.append((unstable_y0_grouped[mu][max_diff_idx + 1] + unstable_y0_grouped[mu][max_diff_idx]) / 2)


    mu_array = np.repeat(mu_list, len(y0_list))
    y_points = np.array(results_flat)
    print(unstable_mu_range, unstable_y_list)
    if coloring:
        sc = ax.scatter(mu_array, y_points, c=np.tile(y0_list, len(mu_list)), cmap='viridis', alpha=0.5, label='Stable numerical')
    else:
        sc = ax.scatter(mu_array, y_points, c='k', alpha=0.5, label='Stable numerical')
    if unstable_mu_range.size != 0:
        ax.scatter(unstable_mu_range, unstable_y_list, color='red', marker='^', label='Unstable numerical')
        ax.vlines(mu_unstable_left, 0, y_max_left, color='red')
        ax.vlines(mu_unstable_right, 0, y_max_right, color='red')
        arrow_length = 0.1 * (y_max_left/2)
        ax.annotate(
            '', 
            xy=(mu_unstable_left, y_max_left/2 + arrow_length / 2), 
            xytext=(mu_unstable_left, y_max_left/2 - arrow_length / 2),
            arrowprops=dict(
                arrowstyle='<-', 
                color='red',
                linewidth=1.5,
                mutation_scale=20
            )
        )
        arrow_length = 0.1 * (y_max_right/2)
        ax.annotate(
            '', 
            xy=(mu_unstable_right, y_max_right/2 + arrow_length / 2), 
            xytext=(mu_unstable_right, y_max_right/2 - arrow_length / 2),
            arrowprops=dict(
                arrowstyle='->', 
                color='red',
                linewidth=1.5,
                mutation_scale=20
            )
        )
    ax.plot(mu_list, np.zeros(len(mu_list)), 'k--')
    ax.set_title(f"t0 = {t0}")
    ax.set_xlabel(r'Amplitude $\mu$')
    ax.set_ylabel(r'Fixed point $y_{fixed}$')
    ax.legend()

    return ax, sc


def create_diagram_hyst(diag_type, t0_list=None, y0_list=None, mu_list=None, a=None, b=None, tau=None, F_type=None, f_type=None, h=None, t_end=None, coloring=False):
    if (t0_list != None) and ((diag_type == 'numerical') or (diag_type == 'comparison')):
        num_plots = len(t0_list)
        ncols = 2 if num_plots > 1 else 1
        nrows = (num_plots + 1) // 2

        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 6))
        fig.suptitle(f"Hysteresis {diag_type} diagram with F - {F_type}, f - {f_type}", fontsize=16)
        if num_plots == 1:
            axs = np.array([axs])
        else:
            axs = np.ravel(axs)
    else:
        fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(10, 6))
        fig.suptitle(f"Hysteresis {diag_type} diagram with F - {F_type}, f - {f_type}", fontsize=16)

    sc = None
    if diag_type == 'numerical':
        for i in range(len(t0_list)):
            axs[i].grid()
            _, sc = create_diagram_hyst_numerical(axs[i], t0_list[i], y0_list, mu_list, a, b, tau, F_type, f_type, h, t_end)
    
    elif diag_type == 'theoretical':
        axs.grid()
        create_diagram_hyst_theoretical(axs, mu_list[-1], a, b, tau, f_type)
    
    elif diag_type == 'comparison':
        for i in range(len(t0_list)):
            axs[i].grid()
            _, sc = create_diagram_hyst_numerical(axs[i], t0_list[i], y0_list, mu_list, a, b, tau, F_type, f_type, h, t_end)
            create_diagram_hyst_theoretical(axs[i], mu_list[-1], a, b, tau, f_type, alpha=0.3)
            
    if ((diag_type == 'numerical') or (diag_type == 'comparison')):
        if coloring:
            cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
            fig.colorbar(sc, cax=cbar_ax, label=r'$y_0$ values')
            
    plt.show()


def ic_fixed(mu, a, b, tau):
    roots = an.compute_roots_t0(mu, a, b, tau)
    y_intercepts = [an.compute_y0(t, mu, b, tau) for t in roots]

    return list(zip(roots, y_intercepts))


def preturbation_eigen(e1, e2, h, F_type, f_type, a, b, mu, tau):
    values = ic_fixed(mu, a, b, tau)[0]
    print(values)
    t0 = -float(values[0])
    y_fixed = float(values[1])

    x0 = 0
    y0 = y_fixed + e1

    x_mapped1, y_mapped1 = nm.runge_kutta_iter(fn.dx_dt, fn.dy_dt, t0, x0, y0, h, F_type, f_type, a, b, mu, tau)
    x0 = e2
    y0 = y_fixed
    x_mapped2, y_mapped2 = nm.runge_kutta_iter(fn.dx_dt, fn.dy_dt, t0, x0, y0, h, F_type, f_type, a, b, mu, tau)

    print(x_mapped1, y_mapped1)
    print(x_mapped2, y_mapped2)
    a = np.array([[0, e1, 0, 0], 
                  [0, 0, 0, e1], 
                  [e2, 0, 0, 0], 
                  [0, 0, e2, 0]])
    b = np.array([x_mapped1,
                  y_mapped1 - y_fixed,
                  x_mapped2,
                  y_mapped2 - y_fixed
])


    v = np.linalg.solve(a, b)

    v = v.reshape(2, 2)
    eigenval, eigenvec = np.linalg.eig(v)

    return eigenval, eigenvec
