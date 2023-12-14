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
from scipy.io import loadmat

# Set figure settings
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import set_fig_dpi, set_style

set_fig_dpi(), set_style()

# %%
# Load the .mat file
mat = loadmat(os.path.join('..', 'data', 'sampleEEGdata.mat'))
# Get the time points
times = mat['EEG'][0][0][14][0]
# Get the epoched data of shape (channels x samples x trials)
data = mat['EEG'][0][0][15]
# Initialize lists for channel names and coordinates
ch_names = []
# Get the list of channel locations
ch_locs = mat['EEG'][0][0][21][0]
# Iterate over the channels
for ch_loc in ch_locs:
    # Append the channel name
    ch_names.append(ch_loc[0][0])

# Get the number of samples and trials
_, n_samples, n_trials = data.shape

# Initialize new data matrices
car_data1 = np.zeros(data.shape)
car_data2 = np.zeros(data.shape)

# Compute the average reference by looping over trials
for trial in range(n_trials):
    for i in range(n_samples):
        # New channel vector is itself minus average over channels
        car_data1[:, i, trial] = data[:, i, trial] - np.mean(data[:, i, trial])

# Compute the average reference in one line
car_data2 = data - np.mean(data, axis=0)

# %% Compare the results

# Convert channel label to index
ch_to_plot = 'POz'
ch_idx = ch_names.index(ch_to_plot)

plt.figure()
plt.plot(times, np.mean(data[ch_idx, :, :], axis=1), label='Earlobe')
plt.plot(times, np.mean(car_data1[ch_idx, :, :], axis=1), label='CAR (loop)')
plt.plot(times, np.mean(car_data2[ch_idx, :, :], axis=1), label='CAR (one line)')

plt.legend()
plt.xlim(-300, 1200)
plt.xlabel('Time (ms)')
plt.ylabel('Voltage ($\mu$V)')
plt.show()

# %%
