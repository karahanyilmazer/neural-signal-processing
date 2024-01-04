# %%
#   COURSE: Neural signal processing and analysis: Zero to hero
#  SESSION: Static spectral analyses
#  TEACHER: Mike X Cohen, sincxpress.com
#    VIDEO: Examples of sharp non-stationarities on power spectra

# !%matplotlib inline
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

# Set figure settings
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import set_fig_dpi, set_style

set_fig_dpi(), set_style(notebook=True)
# %% Sharp transitions

# Parameters
a = [10, 2, 5, 8]
f = [3, 1, 6, 12]
srate = 1000
t = np.arange(0, 10 + 1 / srate, 1 / srate)
n = len(t)

# Sharp transitions
timechunks = np.round(np.linspace(0, n, len(a) + 1)).astype(int)

# Create the signal
signal = np.array([])
for i in range(len(a)):
    chunk = a[i] * np.sin(2 * np.pi * f[i] * t[timechunks[i] : timechunks[i + 1]])
    signal = np.concatenate((signal, chunk))

# Compute its spectrum
signalX = np.fft.fft(signal) / n
hz = np.linspace(0, srate / 2, int(np.floor(n / 2) + 1))

# Plotting
plt.figure()

plt.subplot(2, 1, 1)
plt.plot(t, signal)
plt.xlabel('Time')
plt.ylabel('Amplitude')

plt.subplot(2, 1, 2)
plt.plot(hz, 2 * np.abs(signalX[: len(hz)]), 'o-', markerfacecolor='k')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.xlim([0, 20])

plt.tight_layout()
plt.show()

# %% Edges and edge artifacts
x = (np.linspace(0, 1, n) > 0.5).astype(float)
# Uncommenting this line shows that nonstationarities do not prevent
# stationary signals from being easily observed
# x = x + 0.08 * np.sin(2 * np.pi * 6 * t)

# Plotting
plt.figure()

plt.subplot(2, 1, 1)
plt.plot(t, x)
plt.ylim([-0.1, 1.1])
plt.xlabel('Time (s.)')
plt.ylabel('Amplitude (a.u.)')

plt.subplot(2, 1, 2)
xX = np.fft.fft(x) / n
plt.plot(hz, 2 * np.abs(xX[: len(hz)]))
plt.xlim([0, 20])
plt.ylim([0, 0.1])
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude (a.u.)')

plt.tight_layout()
plt.show()

# %%
