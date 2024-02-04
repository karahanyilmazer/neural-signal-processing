# %%
#   COURSE: Neural signal processing and analysis: Zero to hero
#  SECTION: Time-frequency analysis
#  TEACHER: Mike X Cohen, sincxpress.com
#    VIDEO: Downsampling time-frequency results

# !%matplotlib inline
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

# Set figure settings
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import set_fig_dpi, set_style

set_fig_dpi(), set_style()
# %%
# Take one channel from the v1 dataset, all trials and all time points.
# Extract time-frequency power from 10 to 100 Hz in 42 steps and average over trials.
# Save the full temporal resolution map, and also a version temporally
# downsampled to 40 Hz (one time point each 25 ms [approximate])
# Show images of both maps in the same figure.

# Load the .mat file
mat = loadmat(os.path.join('..', 'data', 'v1_laminar.mat'))
# Get the time points
times = mat['timevec'][0]
# Get the epoched data of shape (channels x samples x trials)
data = mat['csd']
# Get the sampling frequency
srate = mat['srate'][0][0]
# Get the data shape
n_chs, n_pnts, n_trials = data.shape


# These are the time points we want to save
resolution = 0.5
times_to_save = np.arange(-0.3, 1.2 + resolution, resolution)

# Now we need to find those indices in the time vector
t_idx = np.argmin(np.abs(times[:, np.newaxis] - times_to_save), axis=0)

# Soft-coded parameters
freq_range = [10, 100]  # Extract only these frequencies (in Hz)
n_freqs = 42  # Number of frequencies between lowest and highest


# Set up convolution parameters
wave_time = np.arange(-1, 1 - 1 / srate, 1 / srate)
freqs = np.linspace(freq_range[0], freq_range[1], n_freqs)
n_data = data.shape[1] * data.shape[2]
n_kern = len(wave_time)
n_conv = n_data + n_kern - 1
half_wave = (len(wave_time) - 1) // 2


# Create wavelets (do you need to re-create the wavelets? why or why not?)
cmwX = np.zeros((n_freqs, n_conv), dtype=complex)
for fi in range(n_freqs):
    # Create time-domain complex Morlet wavelet
    cmw = np.exp(1j * 2 * np.pi * freqs[fi] * wave_time) * np.exp(
        -4 * np.log(2) * wave_time**2 / 0.3**2
    )

    # Compute fourier coefficients of wavelet and normalize
    cmwX[fi, :] = np.fft.fft(cmw, n_conv)
    cmwX[fi, :] = cmwX[fi, :] / max(cmwX[fi, :])


# %% Initialize time-frequency output matrix
tf = dict()
tf['full'] = np.zeros((n_freqs, n_pnts))
tf['down'] = np.zeros((n_freqs, len(times_to_save)))

# Compute Fourier coefficients of EEG data (doesn't change over frequency!)
eegX = np.fft.fft(data[6, :, :].reshape(-1, order='F'), n_conv)

# Loop over frequencies
for fi in range(n_freqs):
    # Second and third steps of convolution
    comp_sig = np.fft.ifft(eegX * cmwX[fi, :])

    # Cut wavelet back to size of data
    comp_sig = comp_sig[half_wave:-half_wave]
    # Reshape to time X trials
    comp_sig = comp_sig.reshape(n_pnts, n_trials, order='F')

    # Compute power from all time points
    tf['full'][fi, :] = np.mean(np.abs(comp_sig) ** 2, axis=1)

    # Compute power only from the downsampled time points
    tf['down'][fi, :] = np.mean(np.abs(comp_sig[t_idx]) ** 2, 1)


# %% Visualization
fig, axs = plt.subplots(1, 2)
titles = ['Full', 'Downsampled']

axs[0].contourf(times, freqs, tf['full'], 40, vmin=0, vmax=10000)
axs[0].set_title('Full')
axs[0].set_xlim(-0.2, 1.2)
axs[0].set_xlabel('Time (s)')
axs[0].set_ylabel('Frequency (Hz)')

axs[1].contourf(times_to_save, freqs, tf['down'], 40, vmin=0, vmax=10000)
axs[1].set_title('Downsampled')
axs[1].set_xlim(-0.2, 1.2)
axs[1].set_xlabel('Time (s)')

plt.tight_layout()
plt.show()


# Q: Repeat with more downsampling. How low can the new sampling rate be
#    before the data become difficult to interpret?
# A: With this dataset we can go as low as resolution of 0.1 - 0.2 s.


# %%
