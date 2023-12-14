# %%
#     COURSE: Solved problems in neural time series analysis
#    SECTION: Time-domain analyses
#      VIDEO: Compute average reference
# Instructor: sincxpress.com

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

set_fig_dpi(), set_style()
cmap = get_cmap('parula')


# %%
def plot_topomaps(data, t_idx, title, average=False, t_win=None):
    # Define subplot geometry
    n_rows = int(np.ceil(np.sqrt(len(t_idx))))
    n_cols = int(np.ceil(len(t_idx) / n_rows))

    fig, axs = plt.subplots(n_rows, n_cols)
    axs = axs.ravel()
    # Plot the topoplots
    for i in range(len(t_idx)):
        if average:
            # Find the indices of the window
            t_win_idx = round(t_win / (1000 / sfreq))
            # Time points to average together
            times_to_average = np.arange(t_idx[i] - t_win_idx, t_idx[i] + t_win_idx + 1)
            # Average over the data points in the window
            curr_data = np.mean(np.mean(data[:, times_to_average, :], axis=1), axis=1)
        else:
            curr_data = np.mean(data[:, t_idx[i], :], axis=1)

        plot_topomap(
            curr_data,
            info,
            axes=axs[i],
            cmap=cmap,
            vlim=(-10, 10),
            show=False,
        )
        axs[i].set_title(f'{times_to_plot[i]} ms')

    # Disable unused axes
    for ax in axs[len(t_idx) - len(axs) :]:
        ax.axis('off')

    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


# Load the .mat file
mat = loadmat(os.path.join('..', 'data', 'sampleEEGdata.mat'))
# Get the time points
times = mat['EEG'][0][0][14][0]
# Get the epoched data of shape (channels x samples x trials)
data = mat['EEG'][0][0][15]
# Get the sampling frequency
sfreq = mat['EEG'][0][0][11][0][0]

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

# %%
# Time points for topographies
times_to_plot = np.arange(-200, 801, 50)  # in ms

# Convert to indices
t_idx = np.zeros(times_to_plot.shape)
for i, time_point in enumerate(times_to_plot):
    t_idx[i] = int(np.argmin(np.abs(times - time_point)))
t_idx = t_idx.astype(int)

# Alternatively:
# # Convert both arrays to column vectors if they are not already
# EEG_times_col = np.array(times).reshape(-1, 1)
# times_to_plot_row = np.array(times_to_plot).reshape(1, -1)

# # Find the index of the closest time in EEG.times for each time in times_to_plot
# t_idx = np.argmin(np.abs(EEG_times_col - times_to_plot_row), axis=0)


# %%
# Compute the average reference
car_data = data - np.mean(data, axis=0)

# Topoplot time series at exact time point
plot_topomaps(data, t_idx, title='Original Data')
plot_topomaps(car_data, t_idx, title='CAR Data')

# Window size (half of window)
t_win = 10  # in ms

# Topoplot time series at average around time point
plot_topomaps(data, t_idx, title='Averaged Windows', average=True, t_win=t_win)

# %%
