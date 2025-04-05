import numpy as np
import matplotlib.pyplot as plt
import functions as fn
import analytical as an
import num_methods as nm
import ploting as pl
from joblib import Parallel, delayed, parallel_backend
from numba import jit, cuda, float32
import cupy as cp
from functions import dx_dt, dy_dt


def find_first_zero_index(x_values):
    for i in range(1, len(x_values)+1):
        if x_values[-i] == 0:
            return i
    return len(x_values) - 1  # Default case



def compute_y_intercept(t0, y0, mu, a, b, tau, F, f, phi, h, t_end):
    x0 = 0

    _, x_values, y_values, _ = nm.runge_kutta_slide(fn.dx_dt, fn.dy_dt, t0, x0, y0, h, t_end, F, f, a, b, phi, mu, tau, t0_shift=True, period=2*tau)
    y_index = find_first_zero_index(x_values)
    # y_index = np.argmax(x_values == 0)

    if y_values[-y_index-1] >= 0:
        y_intercept = y_values[-y_index-1]
    else:
        y_index_new = np.argmax(x_values[y_index + 1:] == 0)
        y_intercept = y_values[-y_index_new-1]

    return y_intercept



def create_diagram_point(t0_list, y0_list, mu_list, a, b, tau, F, f, phi, h, t_end):
    num_plots = len(t0_list)
    ncols = 2 if num_plots > 1 else 1
    nrows = (num_plots + 1) // 2  # Ensure enough rows

    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 6))
    if num_plots == 1:
        ax = np.array([ax])  # Ensure it's iterable
    else:
        ax = np.ravel(ax)

    for i in range(len(t0_list)):
        with parallel_backend("loky", inner_max_num_threads=1):  # More efficient threading
            results = Parallel(n_jobs=-1, batch_size=10)( 
                delayed(compute_y_intercept)(t0_list[i], y0, mu, a, b, tau, F, f, phi, h, t_end) 
                for mu in mu_list 
                for y0 in y0_list
            )

        mu_array = np.repeat(mu_list, len(y0_list))  # Expand mu values
        y_points = np.array(results)

        sc = ax[i].scatter(mu_array, y_points, c=np.tile(y0_list, len(mu_list)), cmap='viridis', alpha=0.5)
        fig.colorbar(sc, ax=ax[i], label=r'$y_0$ values')  # Attach colorbar to the subplot
        ax[i].plot(mu_list, np.zeros(len(mu_list)), 'k--')
        ax[i].set_title(f"t0 = {t0_list[i]}")
        ax[i].grid()

    plt.tight_layout()
    plt.show()


def preturbation_eigen(e1, e2, t0, h, t_end, F_type, f_type, a, b, phi, mu, tau):
    y_fixed = -an.compute_y0(t0, mu, b, tau)
    x0 = 0
    y0 = y_fixed + e1

    x_mapped1, y_mapped1 = nm.runge_kutta_iter(fn.dx_dt, fn.dy_dt, t0, x0, y0, h, t_end, F_type, f_type, a, b, phi, mu, tau, t0_shift=True, period=2*tau)
    x0 = e2
    y0 = y_fixed
    x_mapped2, y_mapped2 = nm.runge_kutta_iter(fn.dx_dt, fn.dy_dt, t0, x0, y0, h, t_end, F_type, f_type, a, b, phi, mu, tau, t0_shift=True, period=2*tau)

    a = np.array([[0, y_fixed + e1, 0, 0], [0, 0, 0, y_fixed + e1], [e2, y_fixed, 0, 0], [0, 0, e2, y_fixed]])
    b = np.array([0, e1, e2, 0])
    print(b)
    v = np.linalg.solve(a, b)

    v = v.reshape(2, 2)
    eigenval = np.linalg.eig(v)

    return eigenval


# ----------------------------------------------

# @cuda.jit
# def find_first_zero_index_kernel(x_values, y_values, results):
#     i = cuda.grid(1)
#     if i >= x_values.shape[0]:
#         return
    
#     for j in range(x_values.shape[1]):
#         if x_values[i, j] == 0.0:  # Explicitly compare with 0.0
#             results[i] = y_values[i, j]
#             return
    
#     results[i] = y_values[i, -1]  # Default case


# def compute_y_intercept_gpu(t0_list, y0_list, mu_list, a, b, tau, phi, h, t_end):
#     num_points = len(mu_list) * len(y0_list)
    
#     # Transfer input arrays to GPU
#     t0_gpu = cuda.to_device(np.array(t0_list, dtype=np.float64))
#     x0_gpu = cuda.to_device(np.zeros(num_points, dtype=np.float64))
#     y0_gpu = cuda.to_device(np.array(y0_list, dtype=np.float64))
    
#     # Ensure mu_list is correctly passed
#     mu_gpu = cuda.to_device(np.array(mu_list, dtype=np.float64))
    
#     # Allocate memory for results
#     x_values = cuda.device_array((num_points, int(t_end / h) + 1), dtype=np.float64)
#     y_values = cuda.device_array((num_points, int(t_end / h) + 1), dtype=np.float64)
#     results_gpu = cuda.device_array(num_points, dtype=np.float64)
    
#     # CUDA kernel launch parameters
#     threads_per_block = 256
#     blocks_per_grid = (num_points + threads_per_block - 1) // threads_per_block

#     # Call the Runge-Kutta CUDA kernel (pass mu_list correctly)
#     print(type(t0_gpu))
#     print("t0 shape:", t0_gpu.copy_to_host().shape)
#     nm.runge_kutta_kernel[blocks_per_grid, threads_per_block](
#         t0_gpu, x0_gpu, y0_gpu, h, t_end, a, b, phi, mu_gpu, tau, x_values, y_values
#     )
    
#     # Call the kernel to find first zero index
#     find_first_zero_index_kernel[blocks_per_grid, threads_per_block](x_values, y_values, results_gpu)
    
#     return results_gpu.copy_to_host()


# def create_diagram_point_gpu(t0_list, y0_list, mu_list, a, b, tau, F, f, phi, h, t_end):
    
#     num_plots = len(t0_list)
#     fig, ax = plt.subplots(nrows=(num_plots + 1) // 2, ncols=2, figsize=(10, 6))
#     ax = np.ravel(ax) if num_plots > 1 else [ax]
    
#     for i, t0 in enumerate(t0_list):
#         y_points = compute_y_intercept_gpu(t0, y0_list, mu_list, a, b, tau, phi, h, t_end)
        
#         sc = ax[i].scatter(mu_list, y_points, cmap='viridis', alpha=0.5)
#         fig.colorbar(sc, ax=ax[i], label=r'$y_0$ values')
#         ax[i].plot(mu_list, np.zeros(len(mu_list)), 'k--')
#         ax[i].set_title(f"t0 = {t0}")
#         ax[i].grid()
    
#     plt.tight_layout()
#     plt.show()



# # Define your kernel to run on the GPU
# @cuda.jit
# def compute_y_intercept_kernel(t0, mu_list, y0_list, results, a, b, tau, f, F, phi, h, t_end):
#     print(12)
#     idx = cuda.grid(1)  # Each thread processes a (mu, y0) pair
#     if idx < len(mu_list) * len(y0_list):
#         # Retrieve the (mu, y0) pair
#         mu_idx = idx // len(y0_list)
#         y0_idx = idx % len(y0_list)
#         mu = mu_list[mu_idx]
#         y0 = y0_list[y0_idx]

#         # Your computation logic (example of how you can offload)
#         # Initialize variables (example)
#         x0 = 0
#         # Call dx_dt and dy_dt directly (they are now device functions)
#         dx = dx_dt(t0, x0, y0, a, F, f, mu, phi, tau)
#         dy = dy_dt(t0, x0, y0, b, F, f, mu, phi, tau)

#         _, x_values, y_values, _ = nm.runge_kutta_slide(dx, dy, f, F, mu, phi, a, b, tau, 
#                                                         t0, x0, y0, h, t_end, t0_shift=True, period=2*tau)

#         # Find the intercept
#         y_index = np.argmax(x_values == 0)
#         if y_values[-y_index - 1] >= 0:
#             y_intercept = y_values[-y_index - 1]
#         else:
#             y_index_new = np.argmax(x_values[y_index + 1:] == 0)
#             y_intercept = y_values[-y_index_new - 1]
        
#         # Store the result in the output array
#         results[idx] = y_intercept

# def run_cuda_computation(t0, y0_list, mu_list, a, b, tau, dx_dt, dy_dt, f, F, phi, h, t_end):
#     num_results = len(mu_list) * len(y0_list)
#     results = np.zeros(num_results, dtype=np.float32)

#     # Transfer data to device (GPU)
#     d_mu_list = cuda.to_device(mu_list)
#     d_y0_list = cuda.to_device(y0_list)
#     d_results = cuda.to_device(results)

#     # Launch kernel (set block and grid size)
#     threads_per_block = 256
#     blocks_per_grid = (num_results + threads_per_block - 1) // threads_per_block
#     compute_y_intercept_kernel[blocks_per_grid, threads_per_block](t0, d_mu_list, d_y0_list, d_results, a, b, tau, f, F, phi, h, t_end)

#     # Copy results back to CPU
#     results = d_results.copy_to_host()

#     return results


# def create_diagram_point1(t0_list, y0_list, mu_list, dx_dt, dy_dt, f, F, a, b, tau, phi, h, t_end):
#     num_plots = len(t0_list)
#     ncols = 2 if num_plots > 1 else 1
#     nrows = (num_plots + 1) // 2  # Ensure enough rows

#     fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 6))
#     if num_plots == 1:
#         ax = np.array([ax])  # Ensure it's iterable
#     else:
#         ax = np.ravel(ax)

#     for i in range(len(t0_list)):
#         # Use CUDA-enabled function to compute results
#         results = run_cuda_computation(t0_list[i], y0_list, mu_list, a, b, tau, dx_dt, dy_dt, f, F, phi, h, t_end)

#         mu_array = np.repeat(mu_list, len(y0_list))  # Expand mu values
#         y_points = np.array(results)

#         sc = ax[i].scatter(mu_array, y_points, c=np.tile(y0_list, len(mu_list)), cmap='viridis', alpha=0.5)
#         fig.colorbar(sc, ax=ax[i], label=r'$y_0$ values')  # Attach colorbar to the subplot
#         ax[i].plot(mu_list, np.zeros(len(mu_list)), 'k--')
#         ax[i].set_title(f"t0 = {t0_list[i]}")
#         ax[i].grid()

#     plt.tight_layout()
#     plt.show()



# def create_diagram_cycle(t0, y0_list, mu_list, a, b, tau, F, f, phi, h, t_end):
#     x0 = 0

#     y_points = np.array([])
#     for mu in mu_list:
#         f_diag = lambda t: f(t, phi=phi, mu=mu, tau=tau)
#         F_diag = F
#         dx_dt_diag = lambda t, x, y: fn.dx_dt(t, x, y, a, F_diag, f=f_diag)
#         dy_dt_diag = lambda t, x, y: fn.dy_dt(t, x, y, b, F_diag, f=f_diag)
#         for i in range(len(y0_list)):
#             t_values, x_values, y_values, _ = nm.runge_kutta_slide(dx_dt_diag, dy_dt_diag, t0, x0, y0_list[i], h, t_end)
#             np.flip(x_values)
#             y_index = np.argmax(x_values == 0)
#             np.append(y_points, y_values[-y_index-1])

#     fig, ax = plt.subplots()
#     ax.scatter(mu_list, y0_list, c=lambda x, y: y - )
#     plt.grid()