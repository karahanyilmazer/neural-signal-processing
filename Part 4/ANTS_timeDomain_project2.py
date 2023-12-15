# %%
#     COURSE: Solved problems in neural time series analysis
#    SECTION: Time-domain analyses
#      VIDEO: Project 2-2: Solutions
# Instructor: sincxpress.com
#       Goal: Loop through each channel and find the peak time of the ERP between 100
#             and 400 ms. Store these peak times in a separate variable, and then make a
#             topographical plot of the peak times. Repeat for a low-pass filtered ERP.

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
from scipy.signal import filtfilt

# Set figure settings
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import set_fig_dpi, set_style

set_fig_dpi(), set_style()

# %%
# Load the mat file
mat = loadmat(os.path.join('..', 'data', 'sampleEEGdata.mat'))
# Get the time points
times = mat['EEG'][0][0][14][0]
# Get the epoched data of shape (channels x samples x trials)
epoched_data = mat['EEG'][0][0][15]
# Get the sampling frequency
sfreq = mat['EEG'][0][0][11][0][0]

# Get the list of channel names
ch_names = [ch_loc[0][0] for ch_loc in mat['EEG'][0][0][21][0]]

# Create an info object for plotting the topoplot
info = create_info(ch_names, sfreq, 'eeg')
montage = make_standard_montage('standard_1020')
info.set_montage(montage)

# %% Define the filter
low_cut = 15  # in Hz
filt_time = np.arange(-0.3, 0.3, 1 / sfreq)

# Create the sinc function
filt_kern = np.sin(2 * np.pi * low_cut * filt_time) / filt_time

# Replace non-finite values with the maximum finite value
filt_kern[~np.isfinite(filt_kern)] = np.max(filt_kern[np.isfinite(filt_kern)])

# Normalize the filter kernel to unit gain
filt_kern /= np.sum(filt_kern)

# Apply a Hann window
filt_kern *= np.hanning(len(filt_time))

fig, axs = plt.subplots(2, 1)
axs[0].plot(filt_time, filt_kern)
axs[0].set_title('Time Domain')
axs[0].set_xlabel('Time (s)')

hz = np.linspace(0, sfreq, len(filt_kern))
axs[1].plot(hz, np.abs(np.fft.fft(filt_kern)) ** 2, '-o')
axs[1].set_xlim([0, low_cut * 3])
axs[1].set_title('Frequency Domain')
axs[1].set_xlabel('Frequency (Hz)')
axs[1].set_ylabel('Gain')

plt.tight_layout()
plt.show()

# %%
# Define time boundaries and convert to indices
time_range = [100, 400]  # in ms

# Find the indices of the closest times
time_range_idx = np.array([np.argmin(np.abs(times - time)) for time in time_range])
time_range_idx = np.arange(time_range_idx[0], time_range_idx[1] + 1)


# Find the peak times of the ERPs for each channel
erp = np.mean(epoched_data, axis=2)
max_idx = np.argmax(erp[:, time_range_idx], axis=1)
peak_lat = times[time_range_idx[max_idx]]

# Repeat for filtered ERP
erp_filt = filtfilt(filt_kern, 1, erp)
max_idx_filt = np.argmax(erp_filt[:, time_range_idx], axis=1)
peak_lat_filt = times[time_range_idx[max_idx_filt]]

# %%
# Create figure and GridSpec layout
fig = plt.figure()
gs = GridSpec(1, 3, width_ratios=[1, 1, 0.05])

# First and second subplots
ax1 = plt.subplot(gs[0])
ax2 = plt.subplot(gs[1])
ax3 = plt.subplot(gs[2])

# Plot data
plot_topomap(peak_lat, info, axes=ax1, cmap='YlOrBr', vlim=(100, 400), show=False)
im, _ = plot_topomap(
    peak_lat_filt, info, axes=ax2, cmap='YlOrBr', vlim=(100, 400), show=False
)

# Colorbar
fig.colorbar(im, cax=ax3)

fig.suptitle('ERP Peak Times (100-400 ms)', y=0.9, fontsize=17)
ax1.set_title('Raw ERP')
ax2.set_title('Filtered ERP')

plt.tight_layout()
plt.show()

# %%
