# %%
#   COURSE: Neural signal processing and analysis: Zero to hero
#  SESSION: Static spectral analyses
#  TEACHER: Mike X Cohen, sincxpress.com
#    VIDEO: Quantify alpha power over the scalp

# !%matplotlib inline
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
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
mat = loadmat(os.path.join('..', 'data', 'restingstate64chans.mat'))
data = mat['EEG'][0][0][15]
times = mat['EEG'][0][0][14]
srate = mat['EEG'][0][0][11][0][0]
n_samples = mat['EEG'][0][0][10][0][0]

# Get the list of channel locations
ch_locs = mat['EEG'][0][0][21][0]
# Initialize lists for channel names and coordinates
ch_names = []
# Iterate over the channels
for ch_loc in ch_locs:
    # Append the channel name
    ch_names.append(ch_loc[0][0])

# %%
# These data comprise 63 "epochs" of resting-state data. Each epoch is a 2-second
# interval cut from ~2 minutes of resting-state.
# The goal is to compute the power spectrum of each 2-second epoch separately, then
# average together. Then, extract the average power from 8-12 Hz (the "alpha band")
# and make a topographical map of the distribution of this power.

amp = 2 * np.abs(np.fft.fft(data, axis=1) / n_samples)
ch_pwr = amp**2

# Then average over trials
ch_pwr = np.mean(ch_pwr, axis=2)

# Vector of frequencies
hz = np.linspace(0, srate / 2, np.floor(n_samples / 2).astype(int) + 1)

# Plot power spectrum of all channels
plt.figure()
plt.plot(hz, ch_pwr[:, : len(hz)].T)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power ($\mu$V)')
plt.xlim(0, 30)
plt.ylim(0, 50)
plt.show()

# %% Now to extract alpha power

# Boundaries in hz
alpha_bounds = (8, 12)

# Convert to indices
freq_idx = [np.argmin(np.abs(hz - bound)) for bound in alpha_bounds]
freq_idx = np.arange(freq_idx[0], freq_idx[1] + 1)

# Extract average power
alpha_pwr = np.mean(ch_pwr[:, freq_idx], axis=1)

# %%
# Create an info object for plotting the topoplot
info = create_info(ch_names, srate, 'eeg')
montage = make_standard_montage('standard_1020')
info.set_montage(montage)

# Plot the topographic map
_, ax = plt.subplots()
im, _ = plot_topomap(alpha_pwr, info, axes=ax, cmap=cmap, vlim=(0, 6), show=False)
plt.colorbar(im)
plt.title(f'Average Alpha Band Power')
plt.show()

# %%
