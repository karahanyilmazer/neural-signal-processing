# %%
#     COURSE: Solved problems in neural time series analysis
#    SECTION: Time-frequency analyses
#    TEACHER: Mike X Cohen, sincxpress.com
#      VIDEO: Project 4-1: Phase-locked, non-phase-locked, and total power

# !%matplotlib inline
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fft, ifft
from scipy.io import loadmat

# Set figure settings
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import get_cmap, set_fig_dpi, set_style

set_fig_dpi(), set_style()
cmap = get_cmap('parula')

# %% Load v1_laminar.mat data and set parameters
# Load the .mat file
mat = loadmat(os.path.join('..', 'data', 'v1_laminar.mat'))
# Get the time points
time = mat['timevec'][0]
# Get the epoched data of shape (channels x samples x trials)
data = mat['csd']
# Sampling rate
srate = mat['srate'][0, 0]

n_chs, n_pnts, n_trials = data.shape

# %%
ch_idx = 6

# Wavelet parameters
min_freq = 3
max_freq = 80
n_freqs = 40
freqs = np.logspace(np.log10(min_freq), np.log10(max_freq), n_freqs)

# Other wavelet parameters
wave_time = np.arange(-1, 1 + 1 / srate, 1 / srate)
half_wave = len(wave_time) // 2

# Baseline time window
baseline_time = [-0.4, -0.1]

# FFT parameters
n_wave = len(wave_time)
n_data = n_pnts * n_trials
n_conv = n_wave + n_data - 1

# %% Create non-phase-locked dataset

# Compute ERP
erp = np.mean(data[ch_idx, :, :], axis=1)

# Compute induced power by subtracting ERP from each trial
npl = data[ch_idx, :, :] - erp[:, np.newaxis]

# FFT of data
dataX = np.zeros((2, n_conv), dtype=complex)
dataX[0, :] = fft(data[ch_idx, :, :].reshape(-1, order='F'), n_conv)
dataX[1, :] = fft(npl.reshape(-1, order='F'), n_conv)

data1 = fft(data[ch_idx, :, :].reshape(-1, order='F'), n_conv)
data2 = fft(npl.reshape(-1, order='F'), n_conv)

# Convert baseline from ms to indices
base_idx = [np.argmin(np.abs(time - base_time)) for base_time in baseline_time]

# %% Run convolution

# Initialize output time-frequency data
tf = np.zeros((2, len(freqs), n_pnts))

# Loop over frequencies
for fi in range(len(freqs)):
    # Create wavelet
    cmw = np.exp(1j * 2 * np.pi * freqs[fi] * wave_time) * np.exp(
        -4 * np.log(2) * wave_time**2 / 0.2**2
    )

    # Take FFT of wavelet
    cmwX = fft(cmw, n_conv)

    # Run convolution for total and non-phase-locked
    for i in range(2):
        # Convolution
        conv_res = ifft(cmwX * dataX[i, :])
        conv_res = conv_res[half_wave:-half_wave]

        # Reshape back to time X trials
        conv_res = conv_res.reshape(n_pnts, n_trials, order='F')

        # Compute power
        pwr = np.mean(np.abs(conv_res) ** 2, axis=1)

        # dB correct power
        tf[i, fi, :] = 10 * np.log10(pwr / np.mean(pwr[base_idx[0] : base_idx[1]]))

# %% Plotting

# Scale ERP for plotting
erp_scaled = (erp - np.min(erp)) / np.max(erp - np.min(erp))
erp_scaled = erp_scaled * (freqs[-1] - freqs[0]) + freqs[0]
erp_scaled = erp_scaled / 5 + 20

fig, axs = plt.subplots(1, 3, figsize=(15, 6))

# Color limits
clim = np.array([-1, 1]) * 15
levels = np.linspace(clim[0], clim[1], 50)

c = axs[0].contourf(
    time,
    freqs,
    tf[0, :, :],
    40,
    levels=levels,
    extend='max',
    cmap=cmap,
)
axs[0].plot(time, erp_scaled, 'k')  # Plot the ERP on top
axs[0].set_xlim([-0.2, 1.5])
axs[0].set_title('Total')
axs[0].set_xlabel('Time (s)')
axs[0].set_ylabel('Frequency (Hz)')
fig.colorbar(c, ax=axs[0])

c = axs[1].contourf(
    time,
    freqs,
    tf[1, :, :],
    40,
    levels=levels,
    extend='max',
    cmap=cmap,
)
axs[1].plot(time, erp_scaled, 'k')  # Plot the ERP on top
axs[1].set_xlim([-0.2, 1.5])
axs[1].set_title('Non-Phase-Locked')
axs[1].set_xlabel('Time (s)')
fig.colorbar(c, ax=axs[1])

# Phase-locked component is the difference between total and non-phase-locked
c = axs[2].contourf(
    time,
    freqs,
    tf[0, :, :] - tf[1, :, :],
    40,
    levels=levels / 10,
    extend='max',
    cmap=cmap,
)
axs[2].plot(time, erp_scaled, 'k')  # Plot the ERP on top
axs[2].set_xlim([-0.2, 1.5])
axs[2].set_title('Phase-Locked')
axs[2].set_xlabel('Time (s)')
fig.colorbar(c, ax=axs[2])

plt.tight_layout()
plt.show()

# %%
