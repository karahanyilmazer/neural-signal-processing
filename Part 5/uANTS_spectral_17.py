# %%
#   COURSE: Neural signal processing and analysis: Zero to hero
#  SESSION: Static spectral analyses
#  TEACHER: Mike X Cohen, sincxpress.com
#    VIDEO: Welch's method on v1 laminar data

# !%matplotlib inline
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from scipy.signal import welch

# Set figure settings
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import set_fig_dpi, set_style

set_fig_dpi(), set_style(notebook=True)

# %%
# Load the .mat file
mat = loadmat(os.path.join('..', 'data', 'v1_laminar.mat'))
time = mat['timevec'][0]
csd = mat['csd']
srate = mat['srate'][0][0]

# Convert 'csd' to a double precision array (if not already)
csd = np.array(csd, dtype='float64')

# Specify a channel for the analyses
chan = 6  # Python uses 0-based indexing

# Create a Hann window
hannw = 0.5 - np.cos(2 * np.pi * np.linspace(0, 1, csd.shape[1])) / 2

print(hannw.shape)
print(round(csd.shape[1] / 10))

# Welch's method using scipy's welch
hz, pxx = welch(
    csd[chan, :, :],
    fs=srate,
    window=hannw,
    noverlap=round(csd.shape[1] / 10),
    axis=0,
)

# Plotting
plt.figure(figsize=(10, 8))
plt.subplot(2, 1, 1)
plt.plot(time, csd[chan, :, :].mean(axis=1))
plt.xlabel('Time (s)')
plt.ylabel('Voltage (µV)')


plt.subplot(2, 1, 2)
plt.plot(hz, pxx.mean(axis=1))
plt.xlim([0, 140])
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power ($\mu V^2$)')

plt.tight_layout()
plt.show()

# %%
