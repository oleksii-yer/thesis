import numpy as np
import matplotlib.pyplot as plt
from functions import dx_dt, dy_dt
from matplotlib.animation import FuncAnimation


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


def plot_anim(t_values, x_values, y_values, slide, frame_step, include_slide=False, vec_field=False, normed=True):
    # Set up the figure and axis
    fig, ax = plt.subplots()
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Euler Method with Sliding Zone')

    # Plot the x and y axes
    ax.axhline(0, color='black', linewidth=0.8)  # x-axis
    ax.axvline(0, color='black', linewidth=0.8)  # y-axis

    x = np.linspace(-0.2, 0.2, 4)
    y = np.linspace(-5, 5, 30)
    X, Y = np.meshgrid(x, y)

    # Initialize the line object
    line, = ax.plot([], [], 'b', lw=1)
    slide_zone, = ax.plot([], [], 'r')
    if vec_field:
        quiver = ax.quiver(X, Y, np.zeros_like(X), np.zeros_like(Y), scale=25, width=0.004, color='purple')

    # Set limits based on data
    ax.set_xlim(min(x_values), max(x_values))
    ax.set_ylim(min(y_values), max(y_values))

    # Initialization function for the animation
    def init():
        line.set_data([], [])
        slide_zone.set_data([], [])
        return line, slide_zone

    # Update function for the animation
    def update(frame):
        # Update line data up to the current frame
        line.set_data(x_values[:frame], y_values[:frame])

        dx_dt_v = np.vectorize(dx_dt)
        dy_dt_v = np.vectorize(dy_dt)

        if vec_field:
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
        if include_slide:
            res_list.append(slide_zone)
        if vec_field:
            res_list.append(vec_field)

        return res_list

    # Create the animation
    anim = FuncAnimation(fig, update, frames=range(0, len(t_values), frame_step), init_func=init, interval=10, blit=True)
    x = np.arange(-10, 10)
    y = np.arange(-10, 10)
    
    plt.grid()
    plt.close(fig)

    return anim

    # ani.save('animation.gif', writer='pillow', fps=45)
    