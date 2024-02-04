# %%
#   COURSE: Neural signal processing and analysis: Zero to hero
#  SECTION: Time-frequency analysis
#  TEACHER: Mike X Cohen, sincxpress.com
#   VIDEO: Linear vs. logarithmic frequency scaling

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


# Soft-coded parameters
freq_range = [10, 100]  # Extract only these frequencies (in Hz)
n_freqs = 42  # Number of frequencies between lowest and highest

# Select a log or linear frequency scaling
# scale = 'linear'
scale = 'logarithmic'

# Select frequency range
if scale == 'linear':
    freqs = np.linspace(freq_range[0], freq_range[1], n_freqs)
else:
    freqs = np.logspace(np.log10(freq_range[0]), np.log10(freq_range[1]), n_freqs)

# Set up convolution parameters
wave_time = np.arange(-1, 1 - 1 / srate, 1 / srate)
freqs = np.linspace(freq_range[0], freq_range[1], n_freqs)
n_data = data.shape[1] * data.shape[2]
n_kern = len(wave_time)
n_conv = n_data + n_kern - 1
half_wave = (len(wave_time) - 1) // 2


# Create wavelets
cmwX = np.zeros((n_freqs, n_conv), dtype=complex)
for fi in range(n_freqs):
    # Create time-domain complex Morlet wavelet
    cmw = np.exp(1j * 2 * np.pi * freqs[fi] * wave_time) * np.exp(
        -4 * np.log(2) * wave_time**2 / 0.3**2
    )

    # Compute fourier coefficients of wavelet and normalize
    cmwX[fi, :] = np.fft.fft(cmw, n_conv)
    cmwX[fi, :] = cmwX[fi, :] / max(cmwX[fi, :])


# Initialize time-frequency output matrix
tf = np.zeros((n_freqs, n_pnts))

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
    tf[fi, :] = np.mean(np.abs(comp_sig) ** 2, axis=1)


# Visualization in separate figures
plt.figure()
plt.contourf(times, freqs, tf, 40, vmin=0, vmax=10000)
plt.xlim(-0.2, 1.2)
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.title(f'{scale.capitalize()} frequency scaling')
plt.show()


# Q: Is there a difference in the y-axis scaling?
# A: No, the contourf scales the entire graph according to the frequency input.

# %% Manually change the y-axis scaling to log
plt.figure()
plt.contourf(times, freqs, tf, 40, vmin=0, vmax=10000)
plt.xlim(-0.2, 1.2)
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.title(f'{scale.capitalize()} frequency scaling')
if scale == 'logarithmic':
    plt.yscale('log')
    # yticks = freqs[::5]  # Select every 5th frequency
    # yticklabels = [f'{round(freq, 2)}' for freq in yticks]  # Round to 2 decimal places
    # plt.yticks(yticks, yticklabels)

plt.show()


# %%
