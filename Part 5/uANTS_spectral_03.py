# %%
#   COURSE: Neural signal processing and analysis: Zero to hero
#  SESSION: Static spectral analyses
#  TEACHER: Mike X Cohen, sincxpress.com
#    VIDEO: The dot product and sine waves

# !%matplotlib inline
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

# Set figure settings
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import set_fig_dpi, set_style

set_fig_dpi(), set_style(notebook=True)

# %%
# Create two vectors
v1 = np.array([2, 4, 2, 1, 5, 3])
v2 = np.array([4, 2, 2, -3, 2, 5])

# Two (among other) ways to create the dot product
dp1 = np.sum(v1 * v2)
dp2 = np.dot(v1, v2)

print(f'Dot product 1: {dp1}')
print(f'Dot product 2: {dp2}')

# %% In this section, you will create a signal (wavelet)
#    and then compute the dot product between that signal
#    and a series of sine waves.

# Simulation parameters
srate = 1000
time = np.arange(-1, 1 + 1 / srate, 1 / srate)

# Create the signal
theta = 2 * np.pi / 4
signal = np.sin(2 * np.pi * 5 * time + theta) * np.exp((-(time**2)) / 0.1)

# Sine wave frequencies (Hz)
sine_freqs = np.arange(2, 10.5, 0.5)

fig, axs = plt.subplots(2, 1)

dot_prods = np.zeros(len(sine_freqs))
for i in range(len(sine_freqs)):
    # Create a real-valued sine wave with amplitude 1 and phase 0
    sine_wave = np.sin(2 * np.pi * sine_freqs[i] * time)

    # Compute the dot product between sine wave and signal
    # Then normalize by the number of time points
    dot_prods[i] = np.dot(sine_wave, signal) / len(signal)

axs[0].plot(time, signal)
axs[0].set_title('Signal')
axs[0].set_xlabel('Time (s)')
axs[0].set_ylabel('Amplitude (a.u.)')
axs[1].stem(sine_freqs, dot_prods)
axs[1].set_xlabel('Sine Wave Frequency (Hz)')
axs[1].set_ylabel('Dot Product\n(Signed Magnitude)')
axs[1].set_xlim([sine_freqs[0] - 0.5, sine_freqs[-1] + 0.5])
axs[1].set_ylim([-0.2, 0.2])
axs[1].set_title(
    f'Dot Product of Signal and Sine Waves (Offset: {round(theta, 2)} rad.)'
)
plt.tight_layout()
plt.show()

# Q: Try changing the phase. What is the effect on the spectrum of dot products?
# A: Real valued sine waves are not phase invariant.

# %%
