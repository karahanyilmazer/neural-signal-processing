# %%
#   COURSE: Neural signal processing and analysis: Zero to hero
#  SECTION: Time-frequency analysis
#  TEACHER: Mike X Cohen, sincxpress.com
#    VIDEO: Time-frequency trade-off

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
# This code will re-create the plot comparing wavelets with different widths
# in the time and frequency domains.
srate = 512

# Set a few different wavelet widths ("number of cycles" parameter)
n_cycles = [2, 6, 8, 15]
freq = 8

time = np.arange(-2, 2 + 1 / srate, 1 / srate)
hz = np.linspace(0, srate / 2, int(np.floor(len(time) / 2) + 1))

fig, axs = plt.subplots(4, 2)
for i in range(4):
    # Time domain
    plt.subplot(4, 2, i * 2 + 1)
    # Gaussian width as number-of-cycles (don't forget to normalize!)
    s = n_cycles[i] / (2 * np.pi * freq)
    plt.plot(time, np.exp((-(time**2)) / (2 * s**2)))
    plt.title(f'Gaussian with {n_cycles[i]} Cycles')
    if i == 3:
        plt.xlabel('Time (s)')

    # Frequency domain
    plt.subplot(4, 2, i * 2 + 2)
    cmw = np.exp(1j * 2 * np.pi * freq * time) * np.exp((-(time**2)) / (2 * s**2))

    # Take its FFT
    cmwX = np.fft.fft(cmw)
    cmwX = cmwX / max(cmwX)

    # Plot it
    plt.plot(hz, np.abs(cmwX[: len(hz)]))
    plt.xlim(0, 20)
    plt.title(f'Power of Wavelet with {n_cycles[i]} Cycles')
    if i == 3:
        plt.xlabel('Frequency (Hz)')

plt.tight_layout()
plt.show()

# %% comparing wavelet convolution with different wavelet settings
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

# Wavelet parameters
n_freqs = 40
min_freq = 2
max_freq = 30

ch = 'O1'

# Set a few different wavelet widths ("number of cycles" parameter)
n_cycles = [2, 6, 8, 15]

# Time window for baseline normalization
baseline_window = [-500, -200]

# Other wavelet parameters
freqs = np.linspace(min_freq, max_freq, n_freqs)
time = np.arange(-2, 2 + 1 / srate, 1 / srate)
half_wave = int((len(time) - 1) / 2)

# FFT parameters
n_kern = len(time)
n_data = n_samples * n_trials
n_conv = n_kern + n_data - 1

# Initialize output time-frequency data
tf = np.zeros((len(n_cycles), len(freqs), n_samples))

# Convert baseline time into indices
base_idx = np.array([np.argmin(np.abs(times - time)) for time in baseline_window])

# FFT of data (doesn't change on frequency iteration)
dataX = np.fft.fft(data[ch_names.index(ch), :, :].reshape(-1, order='F'), n_conv)

# Loop over cycles
for cycle in range(len(n_cycles)):
    for fi in range(len(freqs)):
        # Create wavelet and get its FFT
        s = n_cycles[cycle] / (2 * np.pi * freqs[fi])

        cmw = np.exp(2 * 1j * np.pi * freqs[fi] * time) * np.exp(
            -(time**2) / (2 * s**2)
        )
        cmwX = np.fft.fft(cmw, n_conv)
        cmwX = cmwX / np.max(cmwX)

        # Run convolution, trim edges, and reshape to 2D (time X trials)
        analytic_signal = np.fft.ifft(cmwX * dataX)
        analytic_signal = analytic_signal[half_wave:-half_wave]
        analytic_signal = analytic_signal.reshape(n_samples, n_trials, order='F')

        # Put power data into big matrix
        tf[cycle, fi, :] = np.mean(np.abs(analytic_signal) ** 2, axis=1)

    # dB normalization
    tf[cycle, :, :] = 10 * np.log10(
        tf[cycle, :, :]
        / np.mean(tf[cycle, :, base_idx[0] : base_idx[1]], axis=1).reshape(-1, 1)
    )

# %% Plot results
fig, axs = plt.subplots(2, 2)
for cycle in range(len(n_cycles)):
    plt.subplot(2, 2, cycle + 1)

    plt.contourf(times, freqs, tf[cycle, :, :], 40, cmap=cmap)
    plt.clim(-3, 3)
    plt.xlim(-300, 1000)
    plt.title(f'Wavelet with {n_cycles[cycle]} Cycles')

axs[1, 0].set_xlabel('Time (ms)')
axs[1, 1].set_xlabel('Time (ms)')
axs[0, 0].set_ylabel('Frequency (Hz)')
axs[1, 0].set_ylabel('Frequency (Hz)')

plt.tight_layout()
plt.show()

# %% Variable number of wavelet cycles

# Set a few different wavelet widths (number of wavelet cycles)
range_cycles = [4, 13]

# Other wavelet parameters
n_cycles = np.logspace(np.log10(range_cycles[0]), np.log10(range_cycles[-1]), n_freqs)

# Initialize output time-frequency data
tf = np.zeros((len(freqs), n_samples))

for fi in range(len(freqs)):
    # Create wavelet and get its FFT
    s = n_cycles[fi] / (2 * np.pi * freqs[fi])
    cmw = np.exp(2 * 1j * np.pi * freqs[fi] * time) * np.exp(
        -(time**2) / (2 * s**2)
    )
    cmwX = np.fft.fft(cmw, n_conv)
    cmwX = cmwX / np.max(cmwX)

    # Run convolution
    analytic_signal = np.fft.ifft(cmwX * dataX)
    analytic_signal = analytic_signal[half_wave:-half_wave]
    analytic_signal = analytic_signal.reshape(n_samples, n_trials, order='F')

    # Put power data into big matrix
    tf[fi, :] = np.mean(np.abs(analytic_signal) ** 2, axis=1)

# db Normalization (we'll talk about this in the next lecture)
tf_db = 10 * np.log10(
    tf / np.mean(tf[:, base_idx[0] : base_idx[1]], axis=1).reshape(-1, 1)
)

# Plot results
plt.figure()

plt.subplot(2, 1, 1)
plt.contourf(times, freqs, tf, 40, cmap=cmap)
plt.clim(0, 5)
plt.xlim(-300, 1000)
plt.title('Convolution with a Range of Cycles')
plt.xlabel('Time (ms)')
plt.ylabel('Frequency (Hz)')
plt.colorbar()

plt.subplot(2, 1, 2)
plt.contourf(times, freqs, tf_db, 40, cmap=cmap)
plt.clim(-3, 3)
plt.xlim(-300, 1000)
plt.title('Same Data But dB Normalized')
plt.xlabel('Time (ms)')
plt.ylabel('Frequency (Hz)')
plt.colorbar()

plt.tight_layout()
plt.show()

# %%
