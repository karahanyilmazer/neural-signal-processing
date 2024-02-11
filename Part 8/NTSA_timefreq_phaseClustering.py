# %%
#     COURSE: Solved problems in neural time series analysis
#    SECTION: Time-frequency analyses
#    TEACHER: Mike X Cohen, sincxpress.com
#      VIDEO: Inter-trial phase clustering before vs. after removing ERP

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

# %% Show Gaussian with different number of cycles
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

# Subtract the ERP from the single trials (to get the Non-Phase Locked activity)

# In a loop
# npl_data = np.zeros(data.shape)
# for chi in range(n_chs):
#     erp = np.mean(data[chi, :, :], axis=1)
#     npl_data[chi, :, :] = data[chi, :, :] - erp[chi, :]

# In one line
erp = np.mean(data, axis=2)
npl_data = data - erp[:, :, np.newaxis]

# %% Plot the difference between original and NPL data
plt.figure()
plt.plot(times, data[30, :, 30], label='Original')
plt.plot(times, npl_data[30, :, 30], label='NPL')
plt.xlabel('Time (ms)')
plt.ylabel('Amplitude')
plt.legend()

plt.show()

# --> The differences between the two lines reflect the ERP (phase-locked activity)

# %% Setup wavelet parameters
min_freq = 2
max_freq = 30
n_freqs = 40
freqs = np.linspace(min_freq, max_freq, n_freqs)

# Which channel to plot
ch = 'O1'
ch_idx = ch_names.index(ch)

# Other wavelet parameters
range_cycles = [4, 10]
s = np.logspace(np.log10(range_cycles[0]), np.log10(range_cycles[1]), n_freqs)
s /= 2 * np.pi * freqs
wave_time = np.arange(-2, 2 + 1 / srate, 1 / srate)
half_wave = (len(wave_time) - 1) // 2

# FFT parameters
n_wave = len(wave_time)
n_data = n_pnts * n_trials
n_conv = n_wave + n_data - 1

# Now compute the FFT of all trials concatenated
all_data = data[ch_idx].reshape(-1, order='F')
dataX = fft(all_data, n_conv)
dataX_sub = fft(npl_data[ch_idx].reshape(-1, order='F'), n_conv)

# Initialize output time-frequency data
tf = np.zeros((2, n_freqs, n_pnts))

# %% Now perform convolution
for fi in range(len(freqs)):
    wavelet = np.exp(2 * 1j * np.pi * freqs[fi] * wave_time) * np.exp(
        -(wave_time**2) / (2 * s[fi] ** 2)
    )
    waveletX = fft(wavelet, n_conv)
    waveletX = waveletX / np.max(waveletX)

    # Convolution for full data
    comp_sig = ifft(waveletX * dataX)
    comp_sig = comp_sig[half_wave:-half_wave]
    comp_sig = comp_sig.reshape(n_pnts, n_trials, order='F')

    # Compute ITPC
    tf[0, fi, :] = np.abs(np.mean(np.exp(1j * np.angle(comp_sig)), axis=1))

    # Repeat for reduced (residual) data
    comp_sig = ifft(waveletX * dataX_sub)
    comp_sig = comp_sig[half_wave:-half_wave]
    comp_sig = comp_sig.reshape(n_pnts, n_trials, order='F')

    # Compute ITPC
    tf[1, fi, :] = np.abs(np.mean(np.exp(1j * np.angle(comp_sig)), axis=1))

# %% Plot the results
erp_label = ['IN', 'EX']

plt.figure()
for i in range(2):
    plt.subplot(2, 1, i + 1)
    plt.contourf(times, freqs, tf[i], 40, cmap=cmap)
    plt.clim(0, 0.7)
    plt.xlim([-500, 1300])
    plt.xlabel('Time (ms)')
    plt.ylabel('Frequency (Hz)')
    plt.title(f'ITPC with ERP {erp_label[i]}cluded')

plt.tight_layout()
plt.show()

# %%
