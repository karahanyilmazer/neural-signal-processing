# %%
#   COURSE: Neural signal processing and analysis: Zero to hero
#  SESSION: Static spectral analyses
#  TEACHER: Mike X Cohen, sincxpress.com
#    VIDEO: The complex dot product

# !%matplotlib qt
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

# Set figure settings
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import set_fig_dpi, set_style

set_fig_dpi(), set_style(notebook=True)
# %%
# Two vectors
v1 = np.array([3j, 4, 5, -3j])
v2 = np.array([-3j, 1j, 1, 0])

# Notice the dot product is a complex number
np.sum(v1 * v2)

# %% Complex dot product with wavelet

# Simulation parameters
srate = 1000
time = np.arange(-1, 1 + 1 / srate, 1 / srate)

# Create the signal
theta = 1 * np.pi / 4
signal = np.sin(2 * np.pi * 5 * time + theta) * np.exp((-(time**2)) / 0.1)

# Sine wave frequencies (Hz)
sine_freqs = np.arange(2, 10.5, 0.5)

fig, axs = plt.subplots(2, 1)

dot_prods = np.zeros(len(sine_freqs), dtype=np.complex64)
for i in range(len(sine_freqs)):
    # Create a complex sine wave with amplitude 1 and phase 0
    csw = np.exp(1j * 2 * np.pi * sine_freqs[i] * time + theta)

    # Compute the dot product between sine wave and signal
    # Then normalize by the number of time points
    dot_prods[i] = np.dot(csw, signal) / len(signal)

axs[0].plot(time, signal)
axs[0].set_title('Signal')
axs[0].set_xlabel('Time (s)')
axs[0].set_ylabel('Amplitude (a.u.)')
axs[1].stem(sine_freqs, np.abs(dot_prods))
axs[1].set_xlabel('Sine Wave Frequency (Hz)')
axs[1].set_ylabel('Dot Product\n(Signed Magnitude)')
axs[1].set_xlim([sine_freqs[0] - 0.5, sine_freqs[-1] + 0.5])
axs[1].set_title(
    f'Dot Product of Signal and Sine Waves (Offset: {round(theta, 2)} rad.)'
)
plt.tight_layout()
plt.show()

# Q: Is the dot product spectrum still dependent on the phase of the signal?
# A: No, because the complex dot product is phase-invariant.
# %% A movie showing why complex sine waves are phase-invariant
# Create complex and real sine wave
time = np.linspace(-1, 1, 1000)
csw = np.exp(1j * 2 * np.pi * 5 * time)
rsw = np.cos(2 * np.pi * 5 * time)

# Specify range of phase offsets for signal
phases = np.linspace(-np.pi / 2, 7 * np.pi / 2, 100)

# Create figure and subplots
fig = plt.figure()
gs = fig.add_gridspec(2, 2)
ax1 = fig.add_subplot(gs[0, 0:2])
ax2 = fig.add_subplot(gs[1, 0])
ax3 = fig.add_subplot(gs[1, 1])

# Adjustments to the plots
ax1.set_xlim([-1, 1])
ax1.set_ylim([-1, 1])
ax1.set_title('Signal and Sine Wave over Time')
ax1.plot(time, rsw, label='Sine Wave')

for ax in [ax2, ax3]:
    ax.set_xlim([-0.2, 0.2])
    ax.set_ylim([-0.2, 0.2])
    ax.grid(True)
    ax.axhline(0, color='k', lw=2)
    ax.axvline(0, color='k', lw=2)
    ax.set_aspect('equal', adjustable='box')

ax2.set_title('Complex Plane')
ax2.set_xlabel('Real Axis')
ax2.set_ylabel('Imag Axis')

ax3.set_title('Real Number Line')
ax3.set_xlabel('Real Axis')
ax3.set_yticklabels([])

# Plot handles for dynamic update
(sig,) = ax1.plot([], [], label='Signal')
(ch,) = ax2.plot([], [], 'ro')
(rh,) = ax3.plot([], [], 'ro')

ax1.legend()


def animate(phi):
    # Create signal
    signal = np.sin(2 * np.pi * 5 * time + phases[phi]) * np.exp((-(time**2)) / 0.1)

    # Compute complex dot product
    cdp = np.sum(signal * csw) / len(time)

    # Compute real-valued dot product
    rdp = np.sum(signal * rsw) / len(time)

    sig.set_data(time, signal)

    # Update complex dot product plot
    ch.set_data([np.real(cdp)], [np.imag(cdp)])

    # Update real dot product plot
    rh.set_data([rdp], [0])

    return ch, rh, sig


# Create animation
ani = FuncAnimation(fig, animate, frames=len(phases), interval=100, blit=True)

plt.tight_layout()
plt.show()
# %%
