# %%
#     COURSE: Solved problems in neural time series analysis
#    SECTION: Time-frequency analyses
#    TEACHER: Mike X Cohen, sincxpress.com
#      VIDEO: Downsampling time-frequency power

# !%matplotlib inline
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fft, ifft
from scipy.io import loadmat

# Set figure settings
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import get_cmap, set_fig_dpi, set_style

set_fig_dpi(), set_style()
cmap = get_cmap('parula')

# %%
# Load the .mat file
mat = loadmat(os.path.join('..', 'data', 'v1_laminar.mat'))
# Get the time points
time = mat['timevec'][0]
# Get the epoched data of shape (channels x samples x trials)
data = mat['csd']
# Sampling rate
srate = mat['srate'][0, 0]

n_chs, n_pnts, n_trials = data.shape

# %%
# Define the time step for downsampling
time_step = 0.25  # in seconds (250 ms = 4 Hz) --> Might be too extreme
time_step = 0.03  # in seconds (30 ms = 33 Hz)

# Vectors of points to extract from TF result
times_to_save = np.arange(-0.2, 1.20 + time_step, time_step)

# Convert that to indices
t_idx = [np.argmin(np.abs(time - t)) for t in times_to_save]

# Baseline time window
base_win = [-0.4, -0.1]
base_idx = [np.argmin(np.abs(time - bw)) for bw in base_win]

# %% Setup wavelet parameters

# Frequency parameters
min_freq = 15
max_freq = 99
n_freqs = 60
frex = np.linspace(min_freq, max_freq, n_freqs)

# Which channel to plot
ch = 6

# Other wavelet parameters
fwhms = np.logspace(np.log10(0.6), np.log10(0.3), n_freqs)
wave_time = np.arange(-2, 2 + 1 / srate, 1 / srate)
half_wave = int(len(wave_time) / 2 - 1)

# FFT parameters
n_wave = len(wave_time)
n_data = n_pnts * n_trials
n_conv = n_wave + n_data - 1

# Now compute the FFT of all trials concatenated
all_data = data[ch].reshape(-1, order='F')
dataX = fft(all_data, n_conv)

# Initialize output time-frequency data
tf_full = np.zeros((n_freqs, len(time)))
tf_down = np.zeros((n_freqs, len(times_to_save)))

# %% Now perform convolution

for fi in range(len(frex)):
    # Create wavelet and get its FFT
    cmw = np.exp(2 * 1j * np.pi * frex[fi] * wave_time) * np.exp(
        -4 * np.log(2) * wave_time**2 / fwhms[fi] ** 2
    )
    cmwX = fft(cmw, n_conv)
    cmwX = cmwX / cmwX[np.argmax(np.abs(cmwX))]

    # Now run convolution in one step
    comp_sig = ifft(cmwX * dataX)
    comp_sig = comp_sig[half_wave + 1 : -half_wave - 1]

    # And reshape back to time X trials
    comp_sig = comp_sig.reshape(len(time), n_trials, order='F')

    # Compute baseline power
    base = np.mean(np.mean(np.abs(comp_sig[base_idx[0] : base_idx[1]]) ** 2, axis=1))

    # Power time series for this frequency, baseline-normalized
    pow_ts = 10 * np.log10(np.mean(np.abs(comp_sig) ** 2, axis=1) / base)

    # Enter full and downsampled data into matrices
    tf_full[fi, :] = pow_ts
    tf_down[fi, :] = pow_ts[t_idx]

# %% Plot results
plt.figure()

clim = [-12, 12]

# Downsampled TF
plt.subplot(221)
plt.contourf(times_to_save, frex, tf_down, 40, cmap=cmap)
plt.clim(clim)
plt.title('Downsampled TF Map')
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.xlim(times_to_save[0], times_to_save[-1])

# Full TF
plt.subplot(222)
plt.contourf(time, frex, tf_full, 40, cmap=cmap)
plt.clim(clim)
plt.title('Full TF Map')
plt.xlabel('Time (s)')
plt.xlim(times_to_save[0], times_to_save[-1])

# Comparison for one frequency
freq_to_plot = 30
f_idx = np.argmin(np.abs(frex - freq_to_plot))

plt.subplot(212)
plt.plot(time, tf_full[f_idx, :], 'k')
plt.plot(times_to_save, tf_down[f_idx, :], 'ro', markerfacecolor='r')
plt.xlim(time[0], time[-1])
plt.legend(['Full Resolution', 'Downsampled'])
plt.xlabel('Time (s)')
plt.ylabel('Power (dB)')

plt.tight_layout()
plt.show()

# %% Compare matrix sizes
sizes = [tf_full.nbytes / 1e6, tf_down.nbytes / 1e6]  # Convert bytes to megabytes

plt.figure(figsize=(5, 4))
plt.bar(['tf_full', 'tf_down'], sizes)
plt.ylabel('Megabytes')

plt.show()

# %%
