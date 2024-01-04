# %%
#   COURSE: Neural signal processing and analysis: Zero to hero
#  SESSION: Static spectral analyses
#  TEACHER: Mike X Cohen, sincxpress.com
#    VIDEO: Sine waves and their parameters

# !%matplotlib inline
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

# Set figure settings
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import set_fig_dpi, set_style

set_fig_dpi(), set_style()
# %% Create a signal by summing together sine waves

# Define a sampling rate
srate = 1000

# List of some frequencies
freqs = [3, 10, 5, 15, 35]

# List of some amplitudes
amps = [5, 15, 10, 5, 7]

# List of some phases between -pi and pi
phases = [np.pi / 7, np.pi / 8, np.pi, np.pi / 2, -np.pi / 4]

# Define time
time = np.arange(-1, 1 + 1 / srate, 1 / srate)

# Loop through frequencies and create sine waves
sine_waves = np.zeros((len(freqs), len(time)))
for i in range(len(amps)):
    sine_waves[i, :] = amps[i] * np.sin(2 * np.pi * time * freqs[i] + phases[i])

# Plot
fig, axs = plt.subplots(len(amps), 1)

for ax, sine_wave in zip(axs, sine_waves):
    ax.plot(time, sine_wave)
    if ax != axs[-1]:
        ax.set_xticks([])

axs[0].set_title('Individual Sine Waves')
plt.xlabel('Time (s)')
plt.tight_layout()
plt.show()

plt.figure()
plt.plot(time, np.sum(sine_waves, axis=0))
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Sum of Sine Waves')
plt.show()

# %%
