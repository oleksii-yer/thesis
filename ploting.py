import numpy as np
import matplotlib.pyplot as plt
from functions import dx_dt, dy_dt
from matplotlib.animation import FuncAnimation
import functions as fn


def plot_basic(t_values, x_values, y_values):
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


def plot_anim(t_values, x_values, y_values, slide, title, frame_step, include_y_intecept=False, include_slide=False, vec_field=False, 
              normed=True, a=None, b=None, f=None, F=None):
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
    y_intercept, = ax.plot([], [], 'ro', markersize=6)
    if vec_field:
        dx_dt = lambda t, x, y: fn.dx_dt(t, x, y, a, F, f)
        dy_dt = lambda t, x, y: fn.dy_dt(t, x, y, b, F, f)
        quiver = ax.quiver(X, Y, np.zeros_like(X), np.zeros_like(Y), scale=25, width=0.004, color='purple')

    # Set limits based on data
    ax.set_xlim(min(x_values), max(x_values))
    ax.set_ylim(min(y_values), max(y_values))

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
            if slide[frame] != 0:
                slide_zone.set_data([0,0], slide[frame])
            else:
                slide_zone.set_data([],[])
            res_list.append(slide_zone)

        if vec_field:
            res_list.append(quiver)

        if include_y_intecept:
            if frame == 0:
                if x_values[frame] == 0:
                    y_intercept.set_data([0], [y_values[frame]])
                else:
                    y_intercept.set_data([], [])
            else:
                if (x_values[frame]*x_values[frame-frame_step] < 0):
                    intercept_frame = np.argmax(x_values[frame-frame_step:frame] == 0) + (frame-frame_step)
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
    