# %%
#!%matplotlib qt
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
# %%
# Load the .mat file
mat = loadmat(os.path.join('..', 'data', 'v1_laminar.mat'))
# Get the time points
times = mat['timevec'][0]
# Get the epoched data of shape (channels x samples x trials)
epoched_data = mat['csd']
# Get the sampling frequency
sfreq = mat['srate'][0][0]

# Get the number of samples
n_samples = epoched_data.shape[1]

tmin, tmax = -500, 1400
# %%
# Calculate the ERP of shape (channels x samples)
erp = np.mean(epoched_data, axis=2)

# Get the index of the channel
ch_idx = 6

# Create an empty figure
fig = plt.figure()
ax = fig.add_subplot()

# Plot the ERP
ax.plot(times*1000, erp[ch_idx, :])
ax.set_title(f'Channel {ch_idx}')
ax.set_xlabel('Time (ms)')
ax.set_ylabel(u'Amplitude (\u03bcA)')
ax.set_xlim(tmin, tmax)
ax.axhline(0, color='gray', ls='--')
ax.axvline(0, color='gray', ls='--')
ax.axvline(500, color='gray', ls='--')
ax.grid()
plt.show()
# %%
# Create an empty figure
fig = plt.figure()
ax = fig.add_subplot()

# Plot depth-by-time image of ERP
ax.contourf(times*1000, np.arange(16)+1, np.squeeze(erp), 40, cmap='jet')
ax.set_xlabel('Time (ms)')
ax.set_ylabel('Cortical Depth')
ax.set_xlim(0, tmax)
plt.show()

# %%
