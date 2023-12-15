# %%
#     COURSE: Solved problems in neural time series analysis
#    SECTION: Time-domain analyses
#      VIDEO: Project 2-1: SOLUTIONS!!
# Instructor: sincxpress.com

# !%matplotlib inline
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import filtfilt

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
epoched_data = mat['EEG'][0][0][15]
# Get the sampling frequency
sfreq = mat['EEG'][0][0][11][0][0]

# Get the list of channel names
ch_names = [ch_loc[0][0] for ch_loc in mat['EEG'][0][0][21][0]]

# Channel to pick
ch_to_use = 'O1'
ch_idx = ch_names.index(ch_to_use)


# Define the time ranges
negpeak_range = [50, 110]
pospeak_range = [110, 170]

# Find the indices of the closest times
negpeak_time = np.array([np.argmin(np.abs(times - time)) for time in negpeak_range])
pospeak_time = np.array([np.argmin(np.abs(times - time)) for time in pospeak_range])

# %% Compute ERP
erp = np.mean(epoched_data[ch_idx, :, :], axis=1)

plt.figure()
plt.plot(times, erp)
plt.title('ERP from Channel ' + ch_to_use)
plt.xlabel('Time (ms)')
plt.ylabel('Voltage ($\mu$V)')
plt.xlim(-300, 1000)

y_min, y_max = plt.ylim()
plt.fill_between(
    times[negpeak_time],
    y_min,
    y_max,
    color=plt.rcParams['axes.prop_cycle'].by_key()['color'][2],
    alpha=0.4,
)
plt.fill_between(
    times[pospeak_time],
    y_min,
    y_max,
    color=plt.rcParams['axes.prop_cycle'].by_key()['color'][3],
    alpha=0.4,
)
plt.ylim(y_min, y_max)

plt.show()

# %% First low-pass filter (windowed sinc function)
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

# %% Now filter the ERP and replot

# Apply filter
erp_filt = filtfilt(filt_kern, 1, erp)

# Plot on top of unfiltered ERP
plt.figure()
plt.plot(times, erp, lw=1)
plt.plot(times, erp_filt)
plt.title('ERP from Channel ' + ch_to_use)
plt.xlabel('Time (ms)')
plt.ylabel('Voltage ($\mu$V)')
plt.xlim(-300, 1000)

y_min, y_max = plt.ylim()
plt.fill_between(
    times[negpeak_time],
    y_min,
    y_max,
    color=plt.rcParams['axes.prop_cycle'].by_key()['color'][2],
    alpha=0.2,
)
plt.fill_between(
    times[pospeak_time],
    y_min,
    y_max,
    color=plt.rcParams['axes.prop_cycle'].by_key()['color'][3],
    alpha=0.2,
)
plt.ylim(y_min, y_max)

plt.show()

# %% Peak-to-peak voltages and timings

# First for unfiltered ERP

# Find minimum/maximum peak times
negpeak_idx_range = np.arange(negpeak_time[0], negpeak_time[1] + 1)
pospeak_idx_range = np.arange(pospeak_time[0], pospeak_time[1] + 1)

erp_min_idx = np.argmin(erp[negpeak_idx_range])
erp_max_idx = np.argmax(erp[pospeak_idx_range])

erp_min_time = times[negpeak_idx_range[erp_min_idx]]
erp_max_time = times[pospeak_idx_range[erp_max_idx]]

# Find minimum/maximum peak values
erp_min_amp = np.min(erp[negpeak_idx_range])
erp_max_amp = np.max(erp[pospeak_idx_range])

# Get results (peak-to-peak voltage and latency)
erp_p2p_amp = np.abs(erp_max_amp - erp_min_amp)
erp_p2p_lat = erp_max_time - erp_min_time


# Then for low-pass filtered ERP

# Find minimum/maximum peak times
erp_filt_min_idx = np.argmin(erp_filt[negpeak_idx_range])
erp_filt_max_idx = np.argmax(erp_filt[pospeak_idx_range])

erp_filt_min_time = times[negpeak_idx_range[erp_filt_min_idx]]
erp_filt_max_time = times[pospeak_idx_range[erp_filt_max_idx]]

# Find minimum/maximum peak values
erp_filt_min_amp = np.min(erp_filt[negpeak_idx_range])
erp_filt_max_amp = np.max(erp_filt[pospeak_idx_range])

# Get results (peak-to-peak voltage and latency)
erp_filt_p2p_amp = np.abs(erp_filt_max_amp - erp_filt_min_amp)
erp_filt_p2p_lat = erp_filt_max_time - erp_filt_min_time

# Report the results
df_dict = {
    'Peak-to-Peak Voltage (µV)': [erp_p2p_amp, erp_filt_p2p_amp],
    'Peak-to-Peak Latency (ms)': [erp_p2p_lat, erp_filt_p2p_lat],
}
df = pd.DataFrame(df_dict, index=['Unfiltered', 'Filtered'])
print('Results Using Single Samples')
print(df)

# %% Repeat for mean around the peak

# Time window for averaging (one-sided)
win = 10  # in ms
# Convert to indices
win_idx = round(win / 1000 * sfreq)

# First for unfiltered ERP

# Find minimum/maximum peak times
negpeak_idx_range = np.arange(negpeak_time[0], negpeak_time[1] + 1)
pospeak_idx_range = np.arange(pospeak_time[0], pospeak_time[1] + 1)

erp_min_idx = np.argmin(erp[negpeak_idx_range])
erp_max_idx = np.argmax(erp[pospeak_idx_range])

min_idx = negpeak_idx_range[erp_min_idx]
max_idx = pospeak_idx_range[erp_max_idx]

erp_min_time = times[min_idx]
erp_max_time = times[max_idx]

# Find minimum/maximum peak values
erp_min_win = np.arange(min_idx - win_idx, min_idx + win_idx + 1)
erp_max_win = np.arange(max_idx - win_idx, max_idx + win_idx + 1)

erp_min_amp = np.mean(erp[erp_min_win])
erp_max_amp = np.mean(erp[erp_max_win])

# Get results (peak-to-peak voltage and latency)
erp_p2p_amp = np.abs(erp_max_amp - erp_min_amp)
erp_p2p_lat = erp_max_time - erp_min_time


# Then for low-pass filtered ERP

# Find minimum/maximum peak times
erp_filt_min_idx = np.argmin(erp_filt[negpeak_idx_range])
erp_filt_max_idx = np.argmax(erp_filt[pospeak_idx_range])

min_idx = negpeak_idx_range[erp_filt_min_idx]
max_idx = pospeak_idx_range[erp_filt_max_idx]

erp_filt_min_time = times[min_idx]
erp_filt_max_time = times[max_idx]

# Find minimum/maximum peak values
erp_filt_min_win = np.arange(min_idx - win_idx, min_idx + win_idx + 1)
erp_filt_max_win = np.arange(max_idx - win_idx, max_idx + win_idx + 1)

erp_filt_min_amp = np.mean(erp_filt[erp_filt_min_win])
erp_filt_max_amp = np.mean(erp_filt[erp_filt_max_win])

# Get results (peak-to-peak voltage and latency)
erp_filt_p2p_amp = np.abs(erp_filt_max_amp - erp_filt_min_amp)
erp_filt_p2p_lat = erp_filt_max_time - erp_filt_min_time


# Report the results
df_dict = {
    'Peak-to-Peak Voltage (µV)': [erp_p2p_amp, erp_filt_p2p_amp],
    'Peak-to-Peak Latency (ms)': [erp_p2p_lat, erp_filt_p2p_lat],
}
df = pd.DataFrame(df_dict, index=['Unfiltered', 'Filtered'])
print('Results Using Mean Windows')
print(df)
# %%
