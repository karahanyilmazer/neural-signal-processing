# %%
#     COURSE: Solved problems in neural time series analysis
#    SECTION: Time-frequency analyses
#    TEACHER: Mike X Cohen, sincxpress.com
#      VIDEO: Visualize time-frequency power from all channels

# !%matplotlib inline
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from mne import create_info
from mne.channels import make_standard_montage
from mne.viz import plot_topomap
from numpy.fft import fft, ifft
from scipy.io import loadmat

# Set figure settings
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import get_cmap, set_fig_dpi, set_style

set_fig_dpi(), set_style()
cmap = get_cmap('parula')

# %% Show Gaussian with different nber of cycles
# Load the .mat file
mat = loadmat(os.path.join('..', 'data', 'sampleEEGdata.mat'))
# Get the time points
time = mat['EEG'][0][0][14][0]
# Get the epoched data of shape (channels x samples x trials)
data = mat['EEG'][0][0][15]
# Get the sampling frequency
srate = mat['EEG'][0][0][11][0][0]

# Get the shape of the data
n_chs, n_pnts, n_trials = data.shape

# Get the list of channel locations
ch_locs = mat['EEG'][0][0][21][0]
# Initialize lists for channel names and coordinates
ch_names = []
ch_loc_xyz = []
ch_loc_theta = []
ch_loc_radius = []
# Iterate over the channels
for ch_loc in ch_locs:
    # Append the channel name
    ch_names.append(ch_loc[0][0])
    # Append the channel coordinate
    ch_loc_xyz.append((ch_loc[3][0][0], ch_loc[4][0][0], ch_loc[5][0][0]))
    ch_loc_theta.append((ch_loc[1][0][0]))
    ch_loc_radius.append((ch_loc[2][0][0]))

# Put the coordinates into an array
ch_loc_xyz = np.array(ch_loc_xyz)
ch_loc_theta = np.array(ch_loc_theta)
ch_loc_radius = np.array(ch_loc_radius)

# Create an info object for plotting the topoplot
info = create_info(ch_names, srate, 'eeg')
montage = make_standard_montage('standard_1020')
info.set_montage(montage)

# %%
# Post-analysis temporal downsampling
times_to_save = np.arange(-200, 801, 25)  # in ms
t_idx = [np.argmin(np.abs(time - t)) for t in times_to_save]

# Baseline window and convert into indices
base_win = [-500, -200]
base_idx = [np.argmin(np.abs(time - bw)) for bw in base_win]

# %% Setup for TF decomposition

# Spectral parameters
min_freq = 3  # Hz
max_freq = 40  # Hz
n_freqs = 30

freqs = np.linspace(min_freq, max_freq, n_freqs)

# Wavelet parameters
fwhms = np.logspace(np.log10(0.6), np.log10(0.3), n_freqs)
wave_time = np.arange(-2, 2 + 1 / srate, 1 / srate)
halfw = (len(wave_time) - 1) // 2

# FFT parameters
n_wave = len(wave_time)
n_data = n_pnts * n_trials
n_conv = n_wave + n_data - 1

# %% Create wavelet spectra

# Initialize
cmwX = np.zeros((n_freqs, n_conv), dtype=complex)

for fi in range(n_freqs):
    cmw = np.exp(1j * 2 * np.pi * freqs[fi] * wave_time) * np.exp(
        -4 * np.log(2) * wave_time**2 / fwhms[fi] ** 2
    )
    tmp = fft(cmw, n_conv)
    cmwX[fi, :] = tmp / np.max(np.abs(tmp))

# %% Time-frequency decomposition

# Initialize matrix
tf = np.zeros((n_chs, n_freqs, len(t_idx)))

for chani in range(n_chs):
    chandat = data[chani].reshape(-1, order='F')
    dataX = fft(chandat, n_conv)

    for fi in range(n_freqs):
        conv_res = ifft(dataX * cmwX[fi, :])
        conv_res = conv_res[halfw:-halfw].reshape(n_pnts, n_trials, order='F')
        avg_pwr = np.mean(np.abs(conv_res) ** 2, axis=1)
        base = np.mean(avg_pwr[base_idx[0] : base_idx[1]])
        tf[chani, fi, :] = 10 * np.log10(avg_pwr[t_idx] / base)

# %% View one channel and topoplot at a time

chan_to_plot = 'Pz'
time_to_plot = 200  # in ms
freq_to_plot = 12  # in Hz

# Find indices
chan_idx = ch_names.index(chan_to_plot)
time_idx = np.argmin(np.abs(times_to_save - time_to_plot))
freq_idx = np.argmin(np.abs(freqs - freq_to_plot))

# Plotting
fig, axs = plt.subplots(1, 2)
c = axs[0].contourf(times_to_save, freqs, tf[chan_idx], 40, cmap='jet')
c.set_clim(-3, 3)
# plt.colorbar()
axs[0].set_title(f'TF Power from Channel {chan_to_plot}')
axs[0].set_xlabel('Time (ms)')
axs[0].set_ylabel('Frequency (Hz)')

# Find the time index closest to 300 ms
t_idx = np.argmin(np.abs(time - time_to_plot))

# Plot the topographic map
im, _ = plot_topomap(
    tf[:, freq_idx, time_idx], info, axes=axs[1], cmap=cmap, show=False
)
plt.title(f'Topo at {time_to_plot} ms & {freq_to_plot} Hz')
plt.show()


# %% Skipping tfviewerx and topoplotIndie due to complexity in direct translation
# For topographical plots in Python, consider using mne.viz.plot_topomap
# TODO: Implement tfviewerx in Python

# %%
