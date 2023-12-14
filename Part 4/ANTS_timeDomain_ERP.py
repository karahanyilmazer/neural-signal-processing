# %%
#     COURSE: Solved problems in neural time series analysis
#    SECTION: Time-domain analyses
#      VIDEO: Event-related potential (ERP)
# Instructor: sincxpress.com

# !%matplotlib inline
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from scipy.io import loadmat

# Set figure settings
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import get_cmap, set_fig_dpi, set_style

set_fig_dpi(), set_style()
cmap = get_cmap('parula')

# %% The theory of an ERP via simulation

# Simulation details
srate = 500
time = np.arange(-1, 2 + 1 / srate, 1 / srate)
n_trials = 100
freq = 9  # Sine wave frequency in Hz
g_peak = 0.43  # Gaussian peak time in seconds
g_width = 0.2  # Gaussian width in seconds

noise_amp = 2  # Noise standard deviation

# Create signal
swave = np.cos(2 * np.pi * freq * time)
gauss = np.exp(-4 * np.log(2) * (time - g_peak) ** 2 / g_width**2)
signal = swave * gauss

# Create data and multiple channels plus noise
data = np.tile(signal, (n_trials, 1))
data = data + noise_amp * np.random.randn(n_trials, len(time))

fig = plt.figure()
gs = GridSpec(5, 1, figure=fig)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[1:4, 0])
ax3 = fig.add_subplot(gs[-1, 0])

ax1.plot(time, signal)
ax1.set_xlim(time[0], time[-1])
ax1.set_ylim(-1.5, 1.5)
ax1.set_title('Pure Signal')
ax1.set_xticklabels([])

ax2.imshow(
    data,
    aspect='auto',
    cmap=cmap,
    vmin=-1 * noise_amp * 2,
    vmax=1 * noise_amp * 2,
    extent=[time[0], time[-1], 0, n_trials],
)
ax2.set_ylabel('Trial')
ax2.set_title('All Trials: Signal + Noise')
ax2.set_xticklabels([])

ax3.plot(time, np.mean(data, axis=0))
ax3.set_xlim(time[0], time[-1])
ax3.set_xlabel('Time (s)')
ax3.set_title('Average over Trials')
ax3.set_ylim(-1.5, 1.5)

plt.tight_layout()
plt.show()

# %% now in real data

data = loadmat(os.path.join('..', 'data', 'v1_laminar.mat'))

time = data['timevec'][0]
csd = data['csd']
ch = 6  # Channel to plot

fig = plt.figure()
gs = GridSpec(3, 1, figure=fig)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[1:, 0])

# Plot ERP from selected channel
ax1.plot(time, np.mean(csd[ch, :, :], axis=1))
ax1.hlines(0, time[0], time[-1], 'grey', '--')
ax1.vlines(0, -1000, 1000, 'grey', '--')
ax1.vlines(0.5, -1000, 1000, 'grey', '--')
ax1.set_title(f'ERP from Channel {ch+1}')
ax1.set_ylabel('Voltage ($\mu V$)')
ax1.set_xlim(time[0], time[-1])


ax2.imshow(
    csd[ch, :, :].T,
    aspect='auto',
    cmap=cmap,
    vmin=-1e3,
    vmax=1e3,
    extent=[time[0], time[-1], 0, csd.shape[1]],
)
ax2.vlines(0, 0, csd.shape[1], 'grey', '--')
ax2.vlines(0.5, 0, csd.shape[1], 'grey', '--')
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Trial')

plt.show()

# %%
plt.figure()

# Make an image of the ERPs from all channels
plt.contourf(time, np.arange(1, 17), np.mean(csd, axis=2), 40, cmap='turbo')
plt.vlines(0, 1, 16, 'grey', '--')
plt.vlines(0.5, 1, 16, 'grey', '--')
plt.title('Time-by-Depth Plot')
plt.xlabel('Time (s)')
plt.ylabel('Channel')

# Reverse the y-axis direction
plt.gca().invert_yaxis()

plt.show()
# %%
