# %%
# !%matplotlib qt
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from mne import create_info
from mne.channels import make_standard_montage
from mne.viz import plot_topomap
from scipy.io import loadmat

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import get_cmap, set_fig_dpi, set_style

# Set figure settings
set_fig_dpi()
set_style()
cmap = get_cmap('parula')

# %%
# Load the .mat file
mat = loadmat(os.path.join('..', 'data', 'sampleEEGdata.mat'))
# Get the time points
times = mat['EEG'][0][0][14][0]
# Get the epoched data of shape (channels x samples x trials)
epoched_data = mat['EEG'][0][0][15]
# Get the sampling frequency
sfreq = mat['EEG'][0][0][11][0][0]

# Get the number of samples
n_samples = epoched_data.shape[1]

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
info = create_info(ch_names, sfreq, 'eeg')
montage = make_standard_montage('standard_1020')
info.set_montage(montage)

tmin, tmax = -400, 1200

# %%
# Scatter plot the channel positions
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(ch_loc_xyz[:, 0], ch_loc_xyz[:, 1], ch_loc_xyz[:, 2], marker='o')

ax.set_title('Sensor Positions')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

plt.show()

# %%
# Calculate the ERP of shape (channels x samples)
erp = np.mean(epoched_data, axis=2)

# Pick a channel to plot the ERP for
ch_to_plot = 'Oz'
# Get the index of the channel
ch_idx = ch_names.index(ch_to_plot)

# Plot the ERP
_, ax = plt.subplots()
ax.plot(times, erp[ch_idx, :])
ax.set_title(f'ERP of Channel {ch_to_plot}')
ax.set_xlabel('Time (ms)')
ax.set_ylabel(u'Amplitude ($\mu A$)')
ax.set_xlim(tmin, tmax)
ax.grid()
plt.show()

# %%
# Pick the time point to plot the topographical map for
time_to_plot = 300  # in ms

# Find the time index closest to 300 ms
t_idx = np.argmin(np.abs(times - time_to_plot))

# Define the lower and upper bound for the colorbar
vmin, vmax = -8, 8

# Plot the topographic map
_, ax = plt.subplots()
im, _ = plot_topomap(erp[:, t_idx], info, axes=ax, cmap=cmap, vlim=(vmin, vmax))
plt.colorbar(im)
plt.title(f'ERP from {time_to_plot} ms')
plt.show()

# %%
