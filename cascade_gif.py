import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# Parameters
n_points = 100  # Number of points in the system
t_max = 7  # Maximum time
dt = 0.1  # Time step
frames = int(t_max / dt)  # Number of frames

# Initial square wave
initial_wave = np.zeros(n_points)
initial_wave[-15:-5] = 1  # Square wave from position -15 to -5

# Discrete-time system: Shift matrix
def update_discrete_wave(wave):
    new_wave = np.zeros_like(wave)
    new_wave[:-1] = wave[1:]
    return new_wave

# Continuous-time system: Smoothing function with diffusion
def update_continuous_wave(wave, alpha):
    new_wave = wave.copy()
    new_wave[1:-1] += alpha * (wave[2:] - 2 * wave[1:-1] + wave[:-2])
    new_wave[0] += alpha * (wave[1] - wave[0])
    new_wave[-1] += alpha * (wave[-2] - wave[-1])
    return new_wave

# Add advection to the continuous-time system
def advect_continuous_wave(wave):
    return np.roll(wave, -1)

# Scaling factor for propagation speed
alpha_cont = 0.1
advect_speed = 1

# Create figure and axes
fig, ax = plt.subplots(figsize=(10, 6))
line1, = ax.plot([], [], label='Discrete Time', color='red', linestyle='dashed')
line2, = ax.plot([], [], label='Continuous Time', color='blue')

# Initialize the plot
def init():
    ax.set_xlim(0, n_points)
    ax.set_ylim(0, 1.5)
    ax.set_xlabel('Hours into the future',fontsize='x-large')
    ax.set_ylabel('Rainfall',fontsize='x-large')
    ax.legend(fontsize='x-large')
    # just under the left end of the x-axis, add a text annotation that says "Now"
    ax.text(-0.03,-0.1,"Now",transform=ax.transAxes,fontsize='x-large')
    # make the x-ticks go from 0 to 6
    plt.xticks(np.arange(0,7,1),fontsize='x-large')

    return line1, line2

# Update the plot for each frame
def update(frame):
    global discrete_wave, continuous_wave, alpha_cont

    # Update the discrete wave (Shift matrix application)
    discrete_wave = update_discrete_wave(discrete_wave)

    # Update the continuous wave (Smoothing function with diffusion and advection)
    continuous_wave = advect_continuous_wave(continuous_wave)
    continuous_wave = update_continuous_wave(continuous_wave, alpha_cont)

    line1.set_data(np.arange(n_points), discrete_wave)
    line2.set_data(np.arange(n_points), continuous_wave)
    return line1, line2

# Initial conditions for the waves
discrete_wave = initial_wave.copy()
continuous_wave = initial_wave.copy()

# Create animation
ani = FuncAnimation(fig, update, frames=frames, init_func=init, blit=True, interval=dt*1000)

# Save animation as GIF
ani.save('square_wave_propagation.gif', writer=PillowWriter(fps=10))

# Display the plot
plt.show()
