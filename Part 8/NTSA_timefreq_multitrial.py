# %%
#     COURSE: Solved problems in neural time series analysis
#    SECTION: Time-frequency analyses
#    TEACHER: Mike X Cohen, sincxpress.com
#      VIDEO: Time-frequency power of multitrial EEG activity

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
# %%
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

# %% Setup wavelet parameters

# Frequency parameters
min_freq = 2
max_freq = 30
num_freqs = 40
freqs = np.linspace(min_freq, max_freq, num_freqs)

# Which channel to plot
ch = 'O1'

# Other wavelet parameters
range_cycles = [4, 10]

s = np.logspace(np.log10(range_cycles[0]), np.log10(range_cycles[1]), num_freqs)
s = s / (2 * np.pi * freqs)
wave_time = np.arange(-2, 2 + 1 / srate, 1 / srate)
half_wave = (len(wave_time) - 1) // 2

# FFT parameters
n_wave = len(wave_time)
n_data = n_pnts * n_trials
n_conv = n_wave + n_data - 1

# Now compute the FFT of all trials concatenated
ch_idx = ch_names.index(ch)
all_data = data[ch_idx, :].reshape(-1, order='F')
dataX = np.fft.fft(all_data, n_conv)

# Initialize output time-frequency data
tf = np.zeros((num_freqs, n_pnts))

# %% Now perform convolution

# Loop over frequencies
for fi in range(len(freqs)):
    # Create wavelet and get its FFT
    wavelet = np.exp(2 * 1j * np.pi * freqs[fi] * wave_time) * np.exp(
        -(wave_time**2) / (2 * s[fi] ** 2)
    )
    waveletX = np.fft.fft(wavelet, n_conv)
    waveletX = waveletX / np.max(waveletX)

    # Now run convolution in one step
    comp_sig = np.fft.ifft(waveletX * dataX)
    comp_sig = comp_sig[half_wave:-half_wave]

    # And reshape back to time X trials
    comp_sig = comp_sig.reshape(n_pnts, n_trials, order='F')

    # Compute power and average over trials
    tf[fi, :] = np.mean(np.abs(comp_sig) ** 2, axis=1)

# Plotting
plt.figure()
plt.contourf(times, freqs, tf, 40, cmap=cmap)
plt.clim([1, 4])
plt.xlim([-500, 1300])
plt.xlabel('Time (ms)')
plt.ylabel('Frequency (Hz)')
plt.show()

# %% Done.
