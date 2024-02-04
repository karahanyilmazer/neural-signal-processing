# %%
#   COURSE: Neural signal processing and analysis: Zero to hero
#  SECTION: Time-frequency analysis
#  TEACHER: Mike X Cohen, sincxpress.com
#    VIDEO: Baseline normalization of TF plots

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

# Get the list of channel locations
ch_locs = mat['EEG'][0][0][21][0]
# Initialize lists for channel names and coordinates
ch_names = []
# Iterate over the channels
for ch_loc in ch_locs:
    # Append the channel name
    ch_names.append(ch_loc[0][0])

n_samples = data.shape[1]
n_trials = data.shape[2]

# Specify baseline periods for dB-normalization
baseline_windows = np.array([[-500, -200], [-100, 0], [0, 300], [-800, 0]])


# Convert baseline time into indices
base_idx = np.argmin(np.abs(times - baseline_windows.reshape(-1, 1)), axis=1).reshape(
    baseline_windows.shape
)

# Setup wavelet parameters

# Frequency parameters
min_freq = 2
max_freq = 30
n_freqs = 40
freqs = np.linspace(min_freq, max_freq, n_freqs)

# Which channel to plot
ch = 'O1'

# Other wavelet parameters
range_cycles = [4, 10]

# Notice: defining cycles as a vector for all frequencies
s = np.logspace(np.log10(range_cycles[0]), np.log10(range_cycles[-1]), n_freqs) / (
    2 * np.pi * freqs
)
wave_time = np.arange(-2, 2 + 1 / srate, 1 / srate)
half_wave = int((len(wave_time) - 1) / 2)

# FFT parameters
n_wave = len(wave_time)
n_data = n_samples * n_trials
n_conv = n_wave + n_data - 1


# Now compute the FFT of all trials concatenated
all_data = data[ch_names.index(ch), :, :].reshape(-1, order='F')
dataX = np.fft.fft(all_data, n_conv)


# Initialize output time-frequency data
tf = np.zeros((len(base_idx), len(freqs), n_samples))

# Now perform convolution

# Loop over frequencies
for fi in range(len(freqs)):
    # Create wavelet and get its FFT
    wavelet = np.exp(2 * 1j * np.pi * freqs[fi] * wave_time) * np.exp(
        -(wave_time**2) / (2 * s[fi] ** 2)
    )
    waveletX = np.fft.fft(wavelet, n_conv)

    # Not necessary because of the decibel normalization. It is only necessary
    # if you want to interpret the raw power in the units of the original data
    # waveletX = waveletX / np.max(waveletX)

    # Now run convolution in one step
    analytic_signal = np.fft.ifft(waveletX * dataX)
    analytic_signal = analytic_signal[half_wave:-half_wave]

    # Reshape back to time X trials
    analytic_signal = analytic_signal.reshape(n_samples, n_trials, order='F')

    # Compute power and average over trials
    tf[3, fi, :] = np.mean(np.abs(analytic_signal) ** 2, axis=1)

# dB normalization and plot results

# Define color limits
clim = [-3, 3]

# Create new matrix for percent change
tfpct = np.zeros(tf.shape)

for base_i in range(tf.shape[0]):
    activity = tf[3, :, :]
    baseline = np.mean(tf[3, :, base_idx[base_i, 0] : base_idx[base_i, 1]], axis=1)

    # Decibel
    tf[base_i, :, :] = 10 * np.log10(activity / baseline.reshape(-1, 1))

# %% Plot results
fig, axs = plt.subplots(2, 2)
axs = axs.ravel()
for i in range(len(axs)):
    cont_plot = axs[i].contourf(times, freqs, tf[i, :, :], 40, cmap=cmap)
    cont_plot.set_clim(clim)
    axs[i].set_xlim(-300, 1000)
    axs[i].set_title(
        f'dB Baseline of {baseline_windows[i, 0]} to {baseline_windows[i, 1]} ms'
    )

axs[2].set_xlabel('Time (ms)')
axs[3].set_xlabel('Time (ms)')
axs[0].set_ylabel('Frequency (Hz)')
axs[2].set_ylabel('Frequency (Hz)')

plt.tight_layout()
plt.show()

# %%
