# %%
#     COURSE: Solved problems in neural time series analysis
#    SECTION: Time-domain analyses
#      VIDEO: Lowpass filter an ERP
# Instructor: sincxpress.com

# !%matplotlib inline
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from scipy.signal import filtfilt, firwin

# Set figure settings
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import set_fig_dpi, set_style

set_fig_dpi(), set_style()

# %%
# Read in the data
data = loadmat(os.path.join('..', 'data', 'v1_laminar.mat'))
csd = data['csd']  # Shape: (channels x time x trials)
srate = data['srate'][0, 0]  # Sampling rate
time_vec = data['timevec'][0]  # Time vector

# Pick one channel to reduce data for convenience
data = csd[6, :, :]  # Shape: (time x trials)

# Cutoff frequency for low-pass filter
low_cut = 20  # Hz

# %% Create and inspect the filter kernel

# Create filter
inc_fac = 18  # Increase filter resolution --> Change to see effect on the filter
filt_ord = round(inc_fac * (low_cut * 1000 / srate))  # Filter order
filt_kern = firwin(filt_ord + 1, low_cut, fs=srate)

# Create time vector
t = np.arange(len(filt_kern)) / srate

# Compute the frequency response
f = np.linspace(0, srate, len(filt_kern))
freq_resp = np.abs(np.fft.fft(filt_kern)) ** 2

# Plot the filter
fig, axs = plt.subplots(2, 1)
axs[0].plot(t, filt_kern)
axs[0].set_xlabel('Time (s)')
axs[0].set_title('Time domain')

axs[1].plot(f, freq_resp, 'o-')
axs[1].set_xlim(0, low_cut * 3)
axs[1].set_xlabel('Frequency (Hz)')
axs[1].set_ylabel('Gain')
axs[1].set_title('Frequency domain')
# axs[1].set_yscale('log')

plt.tight_layout()
plt.show()

# %% Option 1: Compute and filter the ERP

# Extract ERP
erp = np.mean(data, axis=1)

# Apply a zero-phase (non-causal) filter
erp1 = filtfilt(filt_kern, 1, erp)  # (b, a, x) are the inputs --> a=1 for FIR filters

plt.figure()
plt.plot(time_vec, erp, label='Raw ERP')
plt.plot(time_vec, erp1, label='Filtered ERP')
plt.xlabel('Time (s)')
plt.ylabel('Voltage ($\mu$V)')
plt.legend()
plt.show()

# %% Option 2: Filter the single trials then compute ERP

erp2 = np.zeros(time_vec.shape)

for trial in range(data.shape[1]):
    erp2 = erp2 + filtfilt(filt_kern, 1, data[:, trial])

# Complete the averaging
erp2 = erp2 / trial

plt.figure()
plt.plot(time_vec, erp, label='Raw ERP')
plt.plot(time_vec, erp2, label='Filtered ERP')
plt.xlabel('Time (s)')
plt.ylabel('Voltage ($\mu$V)')
plt.legend()
plt.show()

# %% Option 3: Concatenate all trials then filter and compute ERP

# Make one long trial
super_trial = data.T.reshape(-1)

# Apply filter
super_trial = filtfilt(filt_kern, 1, super_trial)

# Reshape back and take average
erp3 = super_trial.reshape(data.T.shape).T
erp3 = np.mean(erp3, axis=1)

plt.figure()
plt.plot(time_vec, erp, label='Raw ERP')
plt.plot(time_vec, erp3, label='Filtered ERP')
plt.xlabel('Time (s)')
plt.ylabel('Voltage ($\mu$V)')
plt.legend()
plt.show()

# %% Now compare all three --> All should be the same except for small edge effects
plt.figure()
plt.plot(time_vec, erp1, label='Filter ERP')
plt.plot(time_vec, erp2, label='Filter trials')
plt.plot(time_vec, erp3, label='Filter concat')

plt.xlabel('Time (s)')
plt.ylabel('Voltage ($\mu$V)')
plt.legend()
plt.show()
# %%
