# %%
#     COURSE: Solved problems in neural time series analysis
#    SECTION: Time-domain analyses
#      VIDEO: Butterfly plot and topo-variance time series
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

# %%
# Compute the average reference
car_data = data - np.mean(data, axis=0)

# Compute the ERPs
erp_ear = np.mean(data, axis=2)
erp_car = np.mean(car_data, axis=2)

# Compute the variance time series for both earlobe and average reference
var_ts_ear = np.var(erp_ear, axis=0)
var_ts_car = np.var(erp_car, axis=0)

# %%
fig, axs = plt.subplots(3, 1)
axs[0].plot(times, erp_ear.T)
axs[0].set_xlim([-500, 1300])
axs[0].set_title('Butterfly Plot (Ear Reference)')
axs[0].set_xlabel('Time (ms)')
axs[0].set_ylabel('Voltage ($\mu$V)')
axs[0].grid()

axs[1].plot(times, erp_car.T)
axs[1].set_xlim([-500, 1300])
axs[1].set_title('Butterfly Plot (Average Reference)')
axs[1].set_xlabel('Time (ms)')
axs[1].set_ylabel('Voltage ($\mu$V)')
axs[1].grid()

axs[2].plot(times, var_ts_ear, label='Earlobe')
axs[2].plot(times, var_ts_car, label='CAR')
axs[2].set_xlim([-500, 1300])
axs[2].set_title('Topographical Variance Time Series')
axs[2].set_xlabel('Time (ms)')
axs[2].set_ylabel('Voltage ($\mu$V)')
axs[2].grid()
axs[2].legend()

plt.tight_layout()
plt.show()

# %%
