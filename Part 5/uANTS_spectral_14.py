# %%
#   COURSE: Neural signal processing and analysis: Zero to hero
#  SESSION: Static spectral analyses
#  TEACHER: Mike X Cohen, sincxpress.com
#    VIDEO: Examples of smooth non-stationarities on power spectra

# !%matplotlib inline
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import detrend

# Set figure settings
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import set_fig_dpi, set_style

set_fig_dpi(), set_style(notebook=True)

# %% Amplitude non-stationarity

# Parameters
srate = 1000
t = np.arange(0, 10 + 1 / srate, 1 / srate)
n = len(t)
f = 3  # Frequency in Hz

# Sine wave with time-increasing amplitude
ampl1 = np.linspace(1, 10, n)
# ampl1 = np.abs(interp1d(np.linspace(t[0], t[-1], 10), 10 * np.random.rand(10))(t))
ampl2 = np.mean(ampl1)

signal1 = ampl1 * np.sin(2 * np.pi * f * t)
signal2 = ampl2 * np.sin(2 * np.pi * f * t)

# Obtain Fourier coefficients and Hz vector
signal1X = np.fft.fft(signal1) / n
signal2X = np.fft.fft(signal2) / n
hz = np.linspace(0, srate, n)

# Plotting
plt.figure()
plt.subplot(2, 1, 1)
plt.plot(t, signal2)
plt.plot(t, signal1)
plt.xlabel('Time')
plt.ylabel('Amplitude')

plt.subplot(2, 1, 2)
plt.plot(hz, 2 * np.abs(signal2X), 'o-', label='Stationary')
plt.plot(hz, 2 * np.abs(signal1X), 's-', label='Non-stationary')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.xlim([1, 7])
plt.legend()

plt.tight_layout()
plt.show()

# %% Frequency non-stationarity
f = [2, 10]
ff = np.linspace(f[0], np.mean(f), n)
signal1 = np.sin(2 * np.pi * ff * t)
signal2 = np.sin(2 * np.pi * np.mean(ff) * t)

signal1X = np.fft.fft(signal1) / n
signal2X = np.fft.fft(signal2) / n
hz = np.linspace(0, srate / 2, int(np.floor(n / 2)))

# Plotting
plt.figure()
plt.subplot(2, 1, 1)
plt.plot(t, signal1)
plt.plot(t, signal2)
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.ylim([-1.1, 1.1])

plt.subplot(2, 1, 2)
plt.plot(hz, 2 * np.abs(signal1X)[: len(hz)], '.-', label='Non-stationary')
plt.plot(hz, 2 * np.abs(signal2X)[: len(hz)], '.-', label='Stationary')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.xlim([0, 20])
plt.legend()

plt.tight_layout()
plt.show()

# %% Examples of rhythmic non-sinusoidal time series
# Parameters
time = np.arange(0, 6, 1 / srate)
n_pnts = len(time)
hz = np.linspace(0, srate, n_pnts)

# Various signals
# Uncomment one signal at a time for testing
signal = detrend(np.sin(np.cos(2 * np.pi * time) - 1))
# signal = np.sin(np.cos(2 * np.pi * time) + time)
# signal = detrend(np.cos(np.sin(2 * np.pi * time)**4))

# Plotting
plt.figure()
plt.subplot(2, 1, 1)
plt.plot(time, signal)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (a.u.)')

plt.subplot(2, 1, 2)
plt.plot(hz, np.abs(np.fft.fft(signal)))
plt.xlim([0, 20])
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude (a.u.)')

plt.tight_layout()
plt.show()
