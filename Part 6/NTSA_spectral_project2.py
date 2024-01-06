# %%
#   COURSE: Solved challenges in neural time series analysis
#  SESSION: Spectral analyses
#  TEACHER: Mike X Cohen, sincxpress.com
#    VIDEO: Project 3-2: Topography of alpha-theta ratio

# !%matplotlib inline
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from mne import create_info
from mne.channels import make_standard_montage
from mne.viz import plot_topomap
from scipy.io import loadmat

# Set figure settings
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import get_cmap, set_fig_dpi, set_style

set_fig_dpi(), set_style(notebook=True)
cmap = get_cmap('parula')

# %%
# Extract alpha/theta ratio for each channel in two time windows:
# [-800 0] and [0 800]. Use trial-specific power. Plot the topographies.
# theta=3-8 Hz, alpha=8-13 Hz. Zero-pad to NFFT=1000

# Load the .mat file
mat = loadmat(os.path.join('..', 'data', 'sampleEEGdata.mat'))
# Get the time points
time = mat['EEG'][0][0][14][0]
# Get the epoched data of shape (channels x samples x trials)
data = mat['EEG'][0][0][15]
# Get the sampling frequency
srate = mat['EEG'][0][0][11][0][0]

# Get the number of samples
n_samples = data.shape[1]

# Get the list of channel locations
ch_locs = mat['EEG'][0][0][21][0]
# Initialize lists for channel names and coordinates
ch_names = []

# Iterate over the channels
for ch_loc in ch_locs:
    # Append the channel name
    ch_names.append(ch_loc[0][0])

# Get the number of channels
n_chs = len(ch_names)

# Create an info object for plotting the topoplot
info = create_info(ch_names, srate, 'eeg')
montage = make_standard_montage('standard_1020')
info.set_montage(montage)

# Timing parameters
mask_pre = np.logical_and((time > -800), (time < 0))  # pre-stimulus
mask_pst = np.logical_and((time > 0), (time < 800))  # post-stimulus
# %%
# FFT parameters
nfft = 1000  # zero-padding!

hz = np.linspace(0, srate, nfft)

# Spectral boundary indices
theta = np.logical_and((hz >= 3), (hz <= 8))
alpha = np.logical_and((hz >= 8), (hz <= 13))

# Extract power

# Obtain Fourier coefficients and extract power spectrum
dataX_pre = np.abs(np.fft.fft(data[:, mask_pre, :], n=nfft, axis=1)) ** 2
dataX_pst = np.abs(np.fft.fft(data[:, mask_pst, :], n=nfft, axis=1)) ** 2

# Band-limited power
theta_pre = np.mean(np.mean(dataX_pre[:, theta, :], axis=1), axis=1)
theta_pst = np.mean(np.mean(dataX_pst[:, theta, :], axis=1), axis=1)

alpha_pre = np.mean(np.mean(dataX_pre[:, alpha, :], axis=1), axis=1)
alpha_pst = np.mean(np.mean(dataX_pst[:, alpha, :], axis=1), axis=1)

# Compute ratios
rat_pre = alpha_pre / theta_pre
rat_pst = alpha_pst / theta_pst


# Define color limits for topoplots
raw_vlim = np.array([-1, 1]) * 0.75 + 1
log_vlim = np.array([-1, 1]) * 0.5

# Plotting
fig, axs = plt.subplots(2, 3)
axs = axs.ravel()

plot_topomap(
    rat_pre,
    info,
    axes=axs[0],
    cmap=cmap,
    vlim=raw_vlim,
    show=False,
)
axs[0].set_title(r'$\alpha / \theta$ PRE')

plot_topomap(
    rat_pst,
    info,
    axes=axs[1],
    cmap=cmap,
    vlim=raw_vlim,
    show=False,
)
axs[1].set_title(r'$\alpha / \theta$ POST')

plot_topomap(
    rat_pst - rat_pre,
    info,
    axes=axs[2],
    cmap=cmap,
    vlim=[-0.2, 0.2],
    show=False,
)
axs[2].set_title(r'$\alpha / \theta$ POST-PRE')

# Repeat for log scaled
plot_topomap(
    np.log10(rat_pre),
    info,
    axes=axs[3],
    cmap=cmap,
    vlim=log_vlim,
    show=False,
)
axs[3].set_title(r'$\log_{10}(\alpha / \theta)$ PRE')

plot_topomap(
    np.log10(rat_pst),
    info,
    axes=axs[4],
    cmap=cmap,
    vlim=log_vlim,
    show=False,
)
axs[4].set_title(r'$\log_{10}(\alpha / \theta)$ POST')

plot_topomap(
    np.log10(rat_pst) - np.log10(rat_pre),
    info,
    axes=axs[5],
    cmap=cmap,
    vlim=log_vlim / 5,
    show=False,
)
axs[5].set_title(r'$\log_{10}(\alpha / \theta)$ POST-PRE')


plt.show()

# %%
