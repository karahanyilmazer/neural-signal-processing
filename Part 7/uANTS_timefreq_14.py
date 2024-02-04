# %%
#   COURSE: Neural signal processing and analysis: Zero to hero
#  SECTION: Time-frequency analysis
#  TEACHER: Mike X Cohen, sincxpress.com
#    VIDEO: Short-time FFT

# !%matplotlib inline
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from scipy.signal import spectrogram
from scipy.signal.windows import hann

# Set figure settings
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import set_fig_dpi, set_style

set_fig_dpi(), set_style()

# %%
# Create signal
srate = 1000
time = np.arange(-3, 3 + 1 / srate, 1 / srate)
pnts = len(time)
freq_mod = np.exp(-(time**2)) * 10 + 10
freq_mod = freq_mod + np.linspace(0, 10, pnts)
signal = np.sin(2 * np.pi * (time + np.cumsum(freq_mod) / srate))

# Plot the signal
fig = plt.figure()
gs = GridSpec(4, 1, figure=fig)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[1:4, 0])

ax1.plot(time, signal)
ax1.set_title('Time-Domain Signal')
ax1.set_ylabel('Amplitude')
ax1.set_xlim([-3, 3])

# Plot the spectrogram
f, t, Sxx = spectrogram(
    signal,
    srate,
    noverlap=150,
    nfft=500,
    window=hann(200),
    scaling='spectrum',
)
ax2.contourf(t - 3, f, np.abs(Sxx), 40, cmap='hot')

ax2.set_ylim([0, 40])
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Frequency (Hz)')

plt.tight_layout()
plt.show()

# %%
# A simplified version to show the mechanics
n = 500
hz = np.linspace(0, srate, n + 1)
tf = np.zeros((pnts // n - 1, len(hz)))
tv = np.zeros((pnts // n - 1))

for i in range(1, pnts // n):
    # Cut some signal
    data_snip = signal[(i * n) - 1 : (i + 1) * n]

    # Compute power in this snippet
    pw = np.abs(np.fft.fft(data_snip)) ** 2
    tf[i - 1, : len(hz)] = pw[: len(hz)]

    # Center time point
    tv[i - 1] = np.mean(time[(i * n) - 1 : (i + 1) * n])

# And plot
fig = plt.figure()
gs = GridSpec(4, 1, figure=fig)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[1:4, 0])

ax1.plot(time, signal)
ax1.set_title('Time-Domain Signal')
ax1.set_xlabel('Time (s)')

ax2.imshow(tf.T, aspect='auto', extent=[tv[0], tv[-1], hz[0], hz[-1]], cmap='hot')
ax2.set_ylim([0, 40])
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Frequency (Hz)')

plt.tight_layout()
plt.show()
# %%
