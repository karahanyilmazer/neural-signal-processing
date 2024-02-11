# %%
#     COURSE: Solved problems in neural time series analysis
#    SECTION: Time-frequency analyses
#    TEACHER: Mike X Cohen, sincxpress.com
#      VIDEO: Project 4-2: Time-frequency power plot via filter-Hilbert

# !%matplotlib inline
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fft, ifft
from scipy.io import loadmat
from scipy.signal import filtfilt, firwin, freqz, hilbert

# Set figure settings
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import get_cmap, set_fig_dpi, set_style

set_fig_dpi(), set_style()
cmap = get_cmap('parula')

# %%
# Load the .mat file
mat = loadmat(os.path.join('..', 'data', 'sampleEEGdata.mat'))
# Get the time points
time = mat['EEG'][0][0][14][0]
# Get the epoched data of shape (channels x samples x trials)
data = mat['EEG'][0][0][15]
# Get the sampling frequency
srate = mat['EEG'][0][0][11][0][0]
nyquist = srate / 2

# Initialize lists for channel names and coordinates
ch_names = []
# Get the list of channel locations
ch_locs = mat['EEG'][0][0][21][0]
# Iterate over the channels
for ch_loc in ch_locs:
    # Append the channel name
    ch_names.append(ch_loc[0][0])

# Get the number of samples and trials
n_chs, n_samples, n_trials = data.shape

# %% TF analysis

# Frequency parameters
n_freqs = 40
freqs = np.linspace(5, 40, n_freqs)
f_width = np.linspace(2, 8, n_freqs)

# Baseline (indices)
base_win = [-500, -200]
base_idx = [np.argmin(np.abs(time - base_time)) for base_time in base_win]

# Strung-out data
ch = 29
all_data = data[ch, :, :].reshape(-1, order='F')

# Initialize output matrices
tf = np.zeros((n_freqs, n_samples))
filt_pwr = np.zeros((n_freqs, 2000))

for fi in range(n_freqs):
    # Create the filter
    freq_range = [freqs[fi] - f_width[fi], freqs[fi] + f_width[fi]]
    filt_order = int(20 * (srate / freq_range[0]))
    filt_order |= 1  # Ensure it's odd
    filt_kern = firwin(
        filt_order + 1,
        freq_range,
        pass_zero=False,
        fs=srate,
    )

    # Compute the power spectrum of the filter kernel
    filt_pwr[fi, :] = np.abs(fft(filt_kern, 2000)) ** 2

    # Apply filter to data
    filt_data = filtfilt(filt_kern, 1, all_data, axis=0)

    # Hilbert transform
    hilb_data = hilbert(filt_data, axis=0)
    hilb_data = hilb_data.reshape(n_samples, n_trials, order='F')

    # Extract power
    pwr = np.abs(hilb_data) ** 2

    # Baseline normalization
    base = np.mean(pwr[base_idx[0] : base_idx[1], :])

    # Trial average and put into TF matrix
    tf[fi, :] = 10 * np.log10(np.mean(pwr, axis=1) / base)

# %% Visualize the filter gains
plt.figure()
plt.imshow(
    filt_pwr,
    aspect='auto',
    extent=[0, srate, freqs[0], freqs[-1]],
    origin='lower',
    cmap=cmap,
)
plt.xlim([0, freqs[-1] + 10])
plt.xlabel('Frequency (Hz)')
plt.ylabel('Filter Center Frequency (Hz)')
plt.show()

# %% Visualize TF matrix
plt.figure()
levels = np.linspace(-3, 3, 40)
c = plt.contourf(time, freqs, tf, 40, levels=levels, cmap=cmap, extend='max')
plt.colorbar(c)
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.xlim([-200, 1300])
plt.show()

# %%
