# %%
#   COURSE: Neural signal processing and analysis: Zero to hero
#  SESSION: Static spectral analyses
#  TEACHER: Mike X Cohen, sincxpress.com
#    VIDEO: Welch's method on phase-slip data

# !%matplotlib inline
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

# Set figure settings
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import set_fig_dpi, set_style

set_fig_dpi(), set_style(notebook=True)
# %%
# Parameters
srate = 1000
time = np.arange(0, srate) / srate

# Create the signal
freq = 10
signal = np.concatenate(
    [np.sin(2 * np.pi * freq * time), np.sin(2 * np.pi * freq * time[::-1])]
)

# Compute the "static" FFT
fft_vals = 2 * np.abs(np.fft.fft(signal)) / len(signal)
hz = np.linspace(0, srate, len(signal))

# Welch's method parameters
win_len = 500  # Window length in points
skip = 100  # Skip in time points

# Vector of frequencies for the small windows
hzL = np.linspace(0, srate / 2, int(np.floor(win_len / 2) + 1))

# Initialize time-frequency matrix (vector in this case)
welch_spect = np.zeros(len(hzL))

# Hann taper
hann = 0.5 * (1 - np.cos(2 * np.pi * np.arange(win_len + 1) / (win_len - 1)))

# Loop over time windows
n_bins = 0
for ti in range(0, len(signal) - win_len, skip):
    # Extract part of the signal
    t_idx = np.arange(ti, ti + win_len + 1)
    tmp_data = signal[t_idx]

    # FFT of these data (does the taper help?)
    x = np.fft.fft(hann * tmp_data) / win_len

    # And put in matrix
    welch_spect = welch_spect + 2 * np.abs(x[: len(hzL)])
    n_bins += 1  # Keep track of how many windows we've added together

# Divide by n_bins to complete average
welch_spect = welch_spect / n_bins

# %%
# Plotting
fig = plt.figure()
gs = GridSpec(2, 2, figure=fig)
ax1 = fig.add_subplot(gs[0, 0:2])
ax2 = fig.add_subplot(gs[1, 0])
ax3 = fig.add_subplot(gs[1, 1])

ax1.plot(signal)
ax1.set_title('Time Domain Signal')
ax1.set_xlabel('Time (ms)')

ax2.bar(hz, fft_vals, width=0.4)
ax2.set_xlim([5, 15])
ax2.set_title('Static FFT')
ax2.set_xlabel('Frequency (Hz)')

ax3.bar(hzL, welch_spect)
ax3.set_xlim([5, 15])
ax3.set_title("Welch's method")
ax3.set_xlabel('Frequency (Hz)')

plt.tight_layout()
plt.show()

# %%
