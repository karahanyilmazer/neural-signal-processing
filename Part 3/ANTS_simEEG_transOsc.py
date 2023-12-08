# %%
#     COURSE: Solved challenges in neural time series analysis
#    SECTION: Simulating EEG data
#      VIDEO: Generating transient oscillations
# Instructor: sincxpress.com

# !%matplotlib qt
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

# Set figure settings
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import set_fig_dpi, set_style

set_fig_dpi(), set_style()

# %% SIMULATION DETAILS

pnts = 4000
srate = 1000
time = np.arange(pnts) / srate - 1

# Gaussian parameters
peak_time = 1  # in seconds
fwhm = 0.4

# Sine wave parameters
sine_freq = 7

# %% CREATE THE SIGNAL

# Create Gaussian taper
gauss = np.exp(-(4 * np.log(2) * (time - peak_time) ** 2) / fwhm**2)

# Sine wave with random phase value ("non-phase-locked")
cosw = np.cos(2 * np.pi * sine_freq * time + 2 * np.pi * np.random.rand())

# Signal
signal = cosw * gauss

# %% PLOTTING
plt.figure()
plt.plot(time, signal)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')

plt.show()

# %%
