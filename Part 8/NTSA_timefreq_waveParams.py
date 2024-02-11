# %%
#     COURSE: Solved problems in neural time series analysis
#    SECTION: Time-frequency analyses
#    TEACHER: Mike X Cohen, sincxpress.com
#      VIDEO: Exploring wavelet parameters in real data

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


# %% Show Gaussian with different number of cycles
# Set different wavelet widths (number of cycles)
num_cycles = [2, 6, 8, 15]
freq = 6.5
time = np.arange(-2, 2 + 1 / srate, 1 / srate)

plt.figure()
for i in range(4):
    plt.subplot(4, 1, i + 1)
    s = num_cycles[i] / (2 * np.pi * freq)  # Number of cycles --> width of Gaussian
    plt.plot(time, np.exp((-(time**2)) / (2 * s**2)))
    plt.title(f'Gaussian with {num_cycles[i]} Cycles')
    plt.ylabel('Amplitude')

plt.xlabel('Time (s)')

plt.tight_layout()
plt.show()

# %% Show frequency spectrum of wavelets with different number of cycles

hz = np.linspace(0, srate / 2, int(np.floor(len(time) / 2) + 1))

plt.figure()
for i in range(4):
    plt.subplot(4, 1, i + 1)

    # Create a Morlet wavelet
    s = num_cycles[i] / (2 * np.pi * freq)
    cmw = np.exp(2 * 1j * np.pi * freq * time) * np.exp(-(time**2) / (2 * s**2))

    # Take its FFT
    cmwX = np.fft.fft(cmw)
    cmwX = cmwX / np.max(cmwX)

    # Plot it
    plt.plot(hz, np.abs(cmwX[: len(hz)]))
    plt.xlim([0, 20])
    plt.title(f'Power of wavelet with {num_cycles[i]} cycles')
    plt.ylabel('Amplitude')

plt.xlabel('Frequency (Hz)')

plt.tight_layout()
plt.show()

# --> Smaller number of cycles has a wider frequency range

# %% Comparing fixed number of wavelet cycles

# Wavelet parameters
num_freqs = 40
min_freq = 2
max_freq = 30
channel2use = 'O1'
baseline_window = [-500, -200]

num_cycles = [2, 6, 8, 15]
freqs = np.linspace(min_freq, max_freq, num_freqs)
time = np.arange(-2, 2 + 1 / srate, 1 / srate)
half_wave = (len(time) - 1) // 2

# FFT parameters
n_kern = len(time)
n_data = n_pnts * n_trials
n_conv = n_kern + n_data - 1

# Initialize output time-frequency data
tf = np.zeros((len(num_cycles), len(freqs), n_pnts))

# Convert baseline time into indices
baseidx = [
    np.argmin(np.abs(times - baseline_window[0])),
    np.argmin(np.abs(times - baseline_window[1])),
]

# FFT of data (doesn't change on frequency iteration)
all_data = data[ch_names.index(channel2use), :, :].reshape(-1, order='F')
dataX = np.fft.fft(all_data, n_conv)

# Loop over cycles
for cyclei in range(len(num_cycles)):
    for fi in range(len(freqs)):
        # Create wavelet and get its FFT
        s = num_cycles[cyclei] / (2 * np.pi * freqs[fi])
        cmw = np.exp(2 * 1j * np.pi * freqs[fi] * time) * np.exp(
            -(time**2) / (2 * s**2)
        )
        cmwX = np.fft.fft(cmw, n_conv)
        cmwX = cmwX / np.max(cmwX)

        # Run convolution, trim edges, and reshape to 2D (time X trials)
        comp_sig = np.fft.ifft(cmwX * dataX)
        comp_sig = comp_sig[half_wave:-half_wave]
        comp_sig = comp_sig.reshape(n_pnts, n_trials, order='F')

        # Put power data into big matrix
        tf[cyclei, fi, :] = np.mean(np.abs(comp_sig) ** 2, axis=1)

    # dB conversion
    baseline_power = np.mean(tf[cyclei, :, baseidx[0] : baseidx[1]], axis=1)
    tf[cyclei, :, :] = 10 * np.log10(tf[cyclei, :, :] / baseline_power[:, None])

# Plot results
plt.figure()
for cyclei in range(len(num_cycles)):
    plt.subplot(2, 2, cyclei + 1)
    contourf_result = plt.contourf(times, freqs, tf[cyclei, :, :], 40, cmap=cmap)
    plt.clim([-3, 3])
    plt.gca().set_xlim([-300, 1000])
    plt.title(f'Wavelet with {num_cycles[cyclei]} Cycles')
    if cyclei > 1:
        plt.xlabel('Time (ms)')
    if cyclei % 2 == 0:
        plt.ylabel('Frequency (Hz)')

plt.tight_layout()
plt.show()

# Time-frequency trade-off:
# Fewer cycles = better temporal resolution, worse frequency resolution
# More cycles = better frequency resolution, worse temporal resolution
# --> Use variable number of cycles:
#     Fewer cycles at low frequencies, more at high frequencies

# %% Variable number of wavelet cycles

# Variable number of cycles: 4 at low frequencies, 13 at high frequencies
range_cycles = [4, 13]
n_cycles = np.logspace(np.log10(range_cycles[0]), np.log10(range_cycles[1]), num_freqs)

# Initialize output time-frequency data
tf = np.zeros((len(freqs), n_pnts))

for fi in range(len(freqs)):
    # Create wavelet and get its FFT
    s = n_cycles[fi] / (2 * np.pi * freqs[fi])
    cmw = np.exp(2 * 1j * np.pi * freqs[fi] * time) * np.exp(
        -(time**2) / (2 * s**2)
    )
    cmwX = np.fft.fft(cmw, n_conv)
    cmwX = cmwX / np.max(cmwX)

    # Run convolution
    comp_sig = np.fft.ifft(cmwX * dataX, n_conv)
    comp_sig = comp_sig[half_wave:-half_wave]
    comp_sig = comp_sig.reshape(n_pnts, n_trials, order='F')

    # Put power data into big matrix
    tf[fi, :] = np.mean(np.abs(comp_sig) ** 2, axis=1)

# dB conversion
tf_db = 10 * np.log10(tf / np.mean(tf[:, baseidx[0] : baseidx[1]], axis=1)[:, None])

# Plot results
vmin, vmax = -3, 3

plt.figure()
levels = np.linspace(vmin, vmax, 100)
c = plt.contourf(times, freqs, tf_db, 40, levels=levels, cmap=cmap, extend='both')
plt.title('TF Power Data with Variable Number of Wavelet Cycles')
plt.xlabel('Time (ms)')
plt.ylabel('Frequency (Hz)')
plt.xlim([-300, 1000])
plt.colorbar(c, label='Power (dB)')
plt.show()

# %%
