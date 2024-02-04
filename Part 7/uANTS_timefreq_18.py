# %%
#   COURSE: Neural signal processing and analysis: Zero to hero
#  SECTION: Time-frequency analysis
#  TEACHER: Mike X Cohen, sincxpress.com
#    VIDEO: Total, non-phase-locked, and phase-locked power
#     GOAL: Here you will separate the non-phase-locked component of the signal
#           to see if they differ. Use the scalp EEG data from channel PO7.

# !%matplotlib inline
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

# Set figure settings
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import get_cmap, set_fig_dpi, set_style

set_fig_dpi(), set_style()
cmap = get_cmap('parula')
# %% Start by loading in the data and picking a channel to work with

# Load the .mat file
mat = loadmat(os.path.join('..', 'data', 'sampleEEGdata.mat'))
# Get the time points
times = mat['EEG'][0][0][14][0]
# Get the epoched data of shape (channels x samples x trials)
data = mat['EEG'][0][0][15]
# Get the sampling frequency
srate = mat['EEG'][0][0][11][0][0]
# Initialize lists for channel names and coordinates
ch_names = []
# Get the list of channel locations
ch_locs = mat['EEG'][0][0][21][0]
# Iterate over the channels
for ch_loc in ch_locs:
    # Append the channel name
    ch_names.append(ch_loc[0][0])

# Get the number of samples and trials
n_chs, n_pnts, n_trials = data.shape

# Pick a channel to plot the ERP for
ch_to_plot = 'PO7'
# Get the index of the channel
ch_idx = ch_names.index(ch_to_plot)

# %%
# How to separate the non-phase-locked signal:
# Start from the assumption that the ERP reflects the phase-locked component.
# This means that the single-trial data contains the
# phase-locked PLUS the non-phase-locked activity.
# Therefore, the non-phase-locked component is obtained by
# subtracting the ERP from the single-trial data.

# But, you need to keep both, so you'll need a separate variable for the
# non-phase-locked signal.
npl_data = data.copy()  # NPL = non-phase-locked

# I'm going to do this over all channels, although the instructions are
# really only for one channel.

for chani in range(n_chs):
    erp = np.mean(data[chani, :, :], axis=1)
    npl_data[chani, :, :] -= erp[:, np.newaxis]

# This can also be done without a loop and without initializing
npl_data = data - np.mean(data, axis=2)[:, :, np.newaxis]

# NOTE: If you are doing this in real data, the non-phase-locked part of
# the signal should be computed separately for each condition. This
# avoids artificial contamination by ERP condition differences.

# Plot the ERP from channel poz for the total and non-phase-locked signal

plt.figure()
plt.plot(times, np.mean(data[ch_idx, :, :], axis=1), label='Total')
plt.plot(times, np.mean(npl_data[ch_idx, :, :], axis=1), label='Non-Phase-Locked')
plt.xlabel('Time (ms)')
plt.ylabel('Activity (μV)')
plt.title('Data.')
plt.xlim([-300, 1200])
plt.legend()

plt.show()

# Q: Are you surprised at the red line? Is it a feature or a bug?!!?
# A: It's a feature. We removed the ERPs from all trials.
#    Then averaging the trials should give us a zero line.


# %% Now for time-frequency analysis
# Apply wavelet convolution to both signal components.
# Extract trial-averaged power and ITPC.
# You can pick the parameters a reasonable frequency range is 2-40 Hz

# Wavelet parameters
n_freqs = 39
min_freq = 2
max_freq = 40

# Parameters
freqs = np.linspace(min_freq, max_freq, n_freqs)
time = np.arange(-1, 1 + 1 / srate, 1 / srate)
half_wave = (len(time) - 1) // 2

# FFT parameters
n_wave = len(time)
n_data = n_pnts * n_trials
n_conv = n_wave + n_data - 1

# Baseline time window
baseidx = [np.argmin(np.abs(times - t)) for t in [-500, -200]]

# FFT of data (all time points, all trials)
dataX_total = np.fft.fft(data[ch_idx, :, :].reshape(-1, order='F'), n_conv)
dataX_npl = np.fft.fft(npl_data[ch_idx, :, :].reshape(-1, order='F'), n_conv)

# Initialize output time-frequency data
# Note: the first '2' is for total/NPL the second '2' is for power/ITPC
tf = np.zeros((2, n_freqs, n_pnts, 2))

# Loop over frequencies
for fi in range(n_freqs):
    # Create wavelet and get its FFT
    fwhm = 0.3
    cmw = np.exp(1j * 2 * np.pi * freqs[fi] * time) * np.exp(
        -4 * np.log(2) * time**2 / fwhm**2
    )
    cmwX = np.fft.fft(cmw, n_conv)

    # Convolution on total activity
    as_total = np.fft.ifft(cmwX * dataX_total)
    as_total = as_total[half_wave:-half_wave]
    as_total = as_total.reshape(n_pnts, n_trials, order='F')

    # Compute power and ITPC
    pwr_total = np.mean(np.abs(as_total) ** 2, axis=1)
    tf[0, fi, :, 0] = 10 * np.log10(
        pwr_total / np.mean(pwr_total[baseidx[0] : baseidx[1]])
    )
    tf[0, fi, :, 1] = np.abs(np.mean(np.exp(1j * np.angle(as_total)), axis=1))

    # Repeat for non-phase-locked activity
    as_npl = np.fft.ifft(cmwX * dataX_npl)
    as_npl = as_npl[half_wave:-half_wave]
    as_npl = as_npl.reshape(n_pnts, n_trials, order='F')

    # Compute power and ITPC
    pwr_npl = np.mean(np.abs(as_npl) ** 2, axis=1)
    tf[1, fi, :, 0] = 10 * np.log10(pwr_npl / np.mean(pwr_npl[baseidx[0] : baseidx[1]]))
    tf[1, fi, :, 1] = np.abs(np.mean(np.exp(1j * np.angle(as_npl)), axis=1))

# %% Now for plotting
# Make a 2x3 grid of imagesc, showing power (top row) and ITPC (bottom row)
# for total (left column) and non-phase-locked (middle column) activity

plt.figure()

# Total power
plt.subplot(231)
plt.contourf(times, freqs, tf[0, :, :, 0], 40, cmap=cmap)
plt.xlim([-300, 1200])
plt.clim([-3, 3])
plt.title('Total Power')
plt.xlabel('Time (ms)')
plt.ylabel('Frequency (Hz)')

# Non-phase-locked power
plt.subplot(232)
plt.contourf(times, freqs, tf[1, :, :, 0], 40, cmap=cmap)
plt.xlim([-300, 1200])
plt.clim([-3, 3])
plt.title('Non-Phase-Locked Power')
plt.xlabel('Time (ms)')

# Total ITPC
plt.subplot(234)
plt.contourf(times, freqs, tf[0, :, :, 1], 40, cmap=cmap)
plt.xlim([-300, 1200])
plt.clim([0, 0.5])
plt.title('Total ITPC')
plt.xlabel('Time (ms)')
plt.ylabel('Frequency (Hz)')

# Non-phase-locked ITPC
plt.subplot(235)
plt.contourf(times, freqs, tf[1, :, :, 1], 40, cmap=cmap)
plt.xlim([-300, 1200])
plt.clim([0, 0.5])
plt.title('Non-Phase-Locked ITPC')
plt.xlabel('Time (ms)')

# Phase-locked power
# Some people compute the phase-locked power as the time-frequency analysis
# of the ERP. Although theoretically sensible, this can be unstable,
# particularly if a baseline normalization is applied.
# A more stable method is to take the difference between total and
# non-phase-locked power.
plt.subplot(233)
plt.contourf(times, freqs, tf[0, :, :, 0] - tf[1, :, :, 0], 40)
plt.xlim([-300, 1200])
plt.clim([-3, 3])
plt.title('Phase-Locked power')
plt.xlabel('Time (ms)')

plt.tight_layout()
plt.show()

# Q: Are you surprised at the non-phase-locked ITPC result?
# A: No, ITPC is already the phase-locked part of the signal.
#    Removing it will give us a map of zeros.
# Q: What features are the same vs. different between the power plots?
# A: Early alpha activity is missing in the non-phase-locked power.
# Q: How does the phase-locked power compare to the total ITPC?
# A: They look similar, as one is the phase-locked power and the 
#    other one is the phase-locking.
# Q: Does it make sense to compute the phase-locked ITPC? Why?
# A: No, as the expected result of the non-phase-locked ITPC is 0.

# %%
