# %%
#     COURSE: Solved challenges in neural time series analysis
#    SECTION: Simulating EEG data
#      VIDEO: Generating "chirps" (frequency-modulated signals)
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
pnts = 10000
srate = 1024
time = np.arange(pnts) / srate

# %% CHIRPS
# Bipolar chirp
freq_bi = np.linspace(5, 15, pnts)

# Multipolar chirp
k = 10  # poles for frequencies
xi = np.arange(k)
yi = np.random.rand(k)
x = np.linspace(0, k - 1, pnts)
y = np.interp(x, xi, yi)
freq_mult = 20 * y

# Signal time series
signal_bi = np.sin(2 * np.pi * ((time + np.cumsum(freq_bi)) / srate))
signal_mult = np.sin(2 * np.pi * ((time + np.cumsum(freq_mult)) / srate))

# %% PLOTTING

mosaic = [['freq', 'freq'], ['chirp', 'chirp']]
fig = plt.figure(constrained_layout=True)
ax_dict = fig.subplot_mosaic(mosaic)

ax_dict['freq'].plot(time, freq_bi)
ax_dict['freq'].set_title('Instantaneous Frequency')
ax_dict['freq'].set_xlabel('Time (s)')
ax_dict['freq'].set_ylabel('Frequency (Hz)')

ax_dict['chirp'].plot(time, signal_bi)
ax_dict['chirp'].set_title('Bipolar Chirp')
ax_dict['chirp'].set_xlabel('Time (s)')
ax_dict['chirp'].set_ylabel('Amplitude (a.u.)')

mosaic = [['freq', 'freq'], ['chirp', 'chirp']]
fig = plt.figure(constrained_layout=True)
ax_dict = fig.subplot_mosaic(mosaic)

ax_dict['freq'].plot(time, freq_mult)
ax_dict['freq'].set_title('Instantaneous Frequency')
ax_dict['freq'].set_xlabel('Time (s)')
ax_dict['freq'].set_ylabel('Frequency (Hz)')

ax_dict['chirp'].plot(time, signal_mult)
ax_dict['chirp'].set_title('Multipolar Chirp')
ax_dict['chirp'].set_xlabel('Time (s)')
ax_dict['chirp'].set_ylabel('Amplitude (a.u.)')

plt.show()

# %%
