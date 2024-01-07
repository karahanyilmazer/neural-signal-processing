# %%
#   COURSE: Neural signal processing and analysis: Zero to hero
#  SECTION: Time-frequency analysis
#  TEACHER: Mike X Cohen, sincxpress.com
#   VIDEO: Time-domain convolution

# !%matplotlib qt
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from scipy.signal import convolve

# Set figure settings
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import set_fig_dpi, set_style

set_fig_dpi(), set_style()
# %% First example to build intuition

# The goal here is to see convolution in a simple example.
# You can try the different kernel options to see how that affects the result.

# Make a kernel (e.g., Gaussian)
kernel = np.exp(-np.linspace(-2, 2, 20) ** 2)
kernel = kernel / np.sum(kernel)

# Try these options
# kernel = -kernel
# kernel = np.concatenate([np.zeros(9), [1, -1], np.zeros(9)])  # edge detector!

# Create the signal
signal = np.concatenate(
    [
        np.zeros(30),
        np.ones(2),
        np.zeros(20),
        np.ones(30),
        2 * np.ones(10),
        np.zeros(30),
        -np.ones(10),
        np.zeros(40),
    ]
)

# Plot
plt.figure()
plt.plot(kernel + 1, 'b', label='Kernel')
plt.plot(signal, 'k', label='Signal')
plt.plot(convolve(signal, kernel, mode='same'), 'r', label='Convolution')
plt.xlim([0, len(signal)])
plt.legend()
plt.show()

# Q: What does the third input in conv() mean? What happens if you remove it?
# A: The third input is the mode of convolution.
#    If you remove it, the result is zero-padded.

# %% A simpler example in more detail

# This cell mainly sets up the animation in the following cell.
# Before moving on to the next cell, try to understand why the result of
# convolution is what it is. Also notice the difference in the y-axis!

# Create signal
signal = np.zeros(20)
signal[7:15] = 1

# Create convolution kernel
kernel = np.array([1, 0.8, 0.6, 0.4, 0.2])

# Convolution sizes
n_signal = len(signal)
n_kernel = len(kernel)
n_conv = n_signal + n_kernel - 1

plt.figure()
# Plot the signal
plt.subplot(311)
plt.plot(signal, 'ko-', markerfacecolor='g', markersize=9)
plt.ylim([-0.1, 1.1])
plt.xlim([-1, n_signal])
plt.title('Signal')

# Plot the kernel
plt.subplot(312)
plt.plot(kernel, 'ko-', markerfacecolor='r', markersize=9)
plt.xlim([-1, n_signal])
plt.ylim([-0.2, 1.1])
plt.title('Kernel')

# Plot the result of convolution
plt.subplot(313)
plt.plot(
    convolve(signal, kernel, mode='same'), 'ko-', markerfacecolor='b', markersize=9
)
plt.xlim([-1, n_signal])
plt.ylim([-0.3, 3.6])
plt.title('Result of Convolution')

plt.tight_layout()
plt.show()

# Q: The kernel has a mean offset. What happens if you mean-center the kernel?
# A: The shape of the convolution result changes.

# %% Convolution in animation

# Movie time parameter
refresh_speed = 0.6  # seconds

# Get the half size of the kernel
half_kern = n_kernel // 2

# Flipped version of kernel
kflip = kernel[::-1]
# kflip = kernel[::-1] - np.mean(kernel) # Mean-centered kernel

# Zero-padded data for convolution
data_to_conv = np.concatenate([np.zeros(half_kern), signal, np.zeros(half_kern)])

# Initialize convolution output
conv_res = np.zeros(n_conv)

# Initialize plot
fig, ax = plt.subplots()
ax.plot(data_to_conv, 'o-', markerfacecolor='g', markersize=9, label='Signal')
(hkern,) = ax.plot(
    [], [], 'o-', markerfacecolor='r', markersize=9, label='Kernel (flip)'
)
(hconv,) = ax.plot(
    [], [], 's-', markerfacecolor='k', markersize=15, label='Convolution'
)
ax.set_title('Animated Convolution')
ax.set_xlim([-1, n_conv])
ax.set_ylim([-3.5, 3.5])
ax.legend(loc='lower right')


# Function to update plot
def update(ti):
    # Get a chunk of data
    tmp_data = data_to_conv[ti - half_kern : ti + half_kern + 1]
    # Compute dot product (don't forget to flip the kernel backwards!)
    conv_res[ti] = np.dot(tmp_data, kflip)

    # Update plot
    hkern.set_data(np.arange(ti - half_kern, ti + half_kern + 1), kflip)
    hconv.set_data(np.arange(half_kern + 1, ti + 1), conv_res[half_kern + 1 : ti + 1])

    return hkern, hconv


# Create animation
ani = FuncAnimation(
    fig,
    update,
    frames=range(half_kern + 1, n_conv - half_kern),
    interval=refresh_speed * 1000,
    blit=True,
)

plt.show()

# %%
