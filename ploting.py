import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import functions as fn
import analytical as an


def plot_basic(x_values, y_values):
    fig, ax = plt.subplots()
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Euler Method with Sliding Zone')

    ax.axhline(0, color='black', linewidth=0.8)  # x-axis
    ax.axvline(0, color='black', linewidth=0.8)  # y-axis

    # Plot the continuous line and dashed segments
    ax.plot(x_values, y_values)
    
    plt.grid()
    plt.show()


def plot_anim(t_values, x_values, y_values, title, frame_step, xlim = None, ylim = None, include_y_intecept=False, include_slide=False, vec_field=False, 
              normed=True, a=None, b=None, mu=None, tau=None, f_type=None, F_type=None):
    # Set up the figure and axis
    fig, ax = plt.subplots()
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(title)

    # Plot the x and y axes
    ax.axhline(0, color='black', linewidth=0.8)  # x-axis
    ax.axvline(0, color='black', linewidth=0.8)  # y-axis

    x = np.linspace(-10, 10, 10)
    y = np.linspace(-2, 2, 4)
    X, Y = np.meshgrid(x, y)

    # Initialize the line object
    line, = ax.plot([], [], 'b', lw=1)
    slide_zone, = ax.plot([], [], 'r')
    y_intercept, = ax.plot([], [], 'go', markersize=4, alpha=0.5)
    if vec_field:
        if f_type == 'meand':
            f = fn.f_meand
        elif f_type == 'sin':
            f = fn.sin
        if F_type == 'closed':
            F = fn.F_closed
        elif F_type == 'open':
            F = fn.F_open
        dx_dt = lambda t, x, y: fn.dx_dt(t, x, y, a, F, f)
        dy_dt = lambda t, x, y: fn.dy_dt(t, x, y, b, F, f)
        quiver = ax.quiver(X, Y, np.zeros_like(X), np.zeros_like(Y), scale=25, width=0.004, color='purple')

    # Set limits based on data
    x_left = np.min(x_values) if xlim == None else xlim[0]
    x_right = np.max(x_values) if xlim == None else xlim[1]
    y_left = np.min(y_values) if ylim == None else ylim[0]
    y_right = np.max(y_values) if ylim == None else ylim[1]

    ax.set_xlim(x_left, x_right)
    ax.set_ylim(y_left, y_right)


    # Initialization function for the animation
    def init():
        line.set_data([], [])
        slide_zone.set_data([], [])
        y_intercept.set_data([], [])
        return line, slide_zone, y_intercept

    intercept_frame = None

    # Update function for the animation
    def update(frame):
        # Update line data up to the current frame
        line.set_data(x_values[:frame], y_values[:frame])

        nonlocal intercept_frame
        nonlocal frame_step

        if vec_field:
            dx_dt_v = np.vectorize(dx_dt)
            dy_dt_v = np.vectorize(dy_dt)

            U = dx_dt_v(t_values[frame], X, Y)
            V = dy_dt_v(t_values[frame], X, Y)
            magnitude = np.sqrt(U**2 + V**2)
            if normed:
                U = U / magnitude
                V = V / magnitude
            quiver.set_UVC(U, V)
        
        res_list = [line]

        if include_slide:
            slide_zone.set_data([0,0], [an.compute_slide_botedge(t_values[frame], a, mu, tau, f_type),
                                        an.compute_slide_topedge(t_values[frame], a, mu, tau, f_type)])
            res_list.append(slide_zone)

        if vec_field:
            res_list.append(quiver)

        if include_y_intecept:
            if (frame == 0) and (x_values[frame] == 0):
                intercept_frame = 0
                y_intercept.set_data([0], [y_values[intercept_frame]])
            else:
                if (len(x_values) - (frame+frame_step)) < 0:
                    frame_step = len(x_values) - frame
                    frame = frame + (frame_step) - 1
                if (x_values[np.max([frame-frame_step, 0])]*x_values[frame] <= 0):
                    intercept_frame = np.argmax(x_values[np.max([frame-frame_step, 0]):frame+1] == 0) + np.max([frame-frame_step, 0])
                if intercept_frame is not None:
                    y_intercept.set_data([0], [y_values[intercept_frame]])
                else:
                    y_intercept.set_data([], [])
            res_list.append(y_intercept)

        return res_list

    # Create the animation
    anim = FuncAnimation(fig, update, frames=range(0, len(t_values), frame_step), init_func=init, interval=10, blit=True)
    x = np.arange(-10, 10)
    y = np.arange(-10, 10)
    
    plt.grid()
    plt.close(fig)

    return anim

    # ani.save('animation.gif', writer='pillow', fps=45)
    