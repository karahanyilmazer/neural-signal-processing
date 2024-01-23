# %%
#   COURSE: Neural signal processing and analysis: Zero to hero
#  SECTION: Time-frequency analysis
#  TEACHER: Mike X Cohen, sincxpress.com
#    VIDEO: Inter-trial phase clustering (ITPC/ITC)

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
# ITPC with different variances

# The goal here is to develop some visual intuition for the
#   correspondence between ITPC and distributions of phase angles.

# Specify parameters
circ_prop = 0.5  # proportion of the circle to fill
N = 100  # number of "trials"

# Generate phase angle distribution
sim_data = np.random.rand(N) * (2 * np.pi) * circ_prop

# Compute ITPC and preferred phase angle
itpc = np.abs(np.mean(np.exp(1j * sim_data)))
pref_angle = np.angle(np.mean(np.exp(1j * sim_data)))

# Plotting
plt.figure()

# as linear histogram
plt.subplot(121)
plt.hist(sim_data, 20)
plt.xlabel('Phase angle')
plt.ylabel('Count')
plt.xlim(0, 2 * np.pi)
plt.title(f'Observed ITPC: {itpc:.4f}')

# and as polar distribution
plt.subplot(122, projection='polar')
plt.polar([np.zeros(N), sim_data], [np.zeros(N), np.ones(N)], 'grey')
plt.polar([0, pref_angle], [0, itpc], 'r')

plt.show()

# %% Compute and plot TF-ITPC for one electrode
# Load the .mat file
mat = loadmat(os.path.join('..', 'data', 'sampleEEGdata.mat'))
# Get the time points
times = mat['EEG'][0][0][14][0]
# Get the epoched data of shape (channels x samples x trials)
data = mat['EEG'][0][0][15]
# Get the sampling frequency
srate = mat['EEG'][0][0][11][0][0]

# Get the list of channel locations
ch_locs = mat['EEG'][0][0][21][0]
# Initialize lists for channel names and coordinates
ch_names = []
# Iterate over the channels
for ch_loc in ch_locs:
    # Append the channel name
    ch_names.append(ch_loc[0][0])

# Wavelet parameters
n_freqs = 40
min_freq = 2
max_freq = 30

ch = 'Pz'

# Set range for variable number of wavelet cycles
range_cycles = [3, 10]

# Parameters (notice using logarithmically spaced frequencies!)
freqs = np.logspace(np.log10(min_freq), np.log10(max_freq), n_freqs)
n_cycs = np.logspace(np.log10(range_cycles[0]), np.log10(range_cycles[-1]), n_freqs)
time = np.arange(-2, 2 + 1 / srate, 1 / srate)
half_wave = int((len(time) - 1) / 2)

# Get the number of samples
n_samples = data.shape[1]
n_trials = data.shape[2]

# FFT parameters
n_wave = len(time)
n_data = n_samples * n_trials
n_conv = n_wave + n_data - 1

# FFT of data (doesn't change on frequency iteration)
dataX = np.fft.fft(data[ch_names.index(ch), :, :].reshape(-1, order='F'), n_conv)

# Initialize output time-frequency data
tf = np.zeros((n_freqs, n_samples))

# Loop over frequencies
for fi in range(n_freqs):
    # Create wavelet and get its FFT
    s = n_cycs[fi] / (2 * np.pi * freqs[fi])
    wavelet = np.exp(2 * 1j * np.pi * freqs[fi] * time) * np.exp(
        -(time**2) / (2 * s**2)
    )
    waveletX = np.fft.fft(wavelet, n_conv)

    # Question: is this next line necessary?
    waveletX = waveletX / np.max(waveletX)

    # Run convolution
    analytic_signal = np.fft.ifft(waveletX * dataX, n_conv)
    analytic_signal = analytic_signal[half_wave:-half_wave]

    # Reshape back to time X trials
    analytic_signal = analytic_signal.reshape(n_samples, n_trials, order='F')

    # Compute ITPC
    tf[fi, :] = np.abs(np.mean(np.exp(1j * np.angle(analytic_signal)), axis=1))

# Plot results
plt.figure()
plt.contourf(times, freqs, tf, 40)
plt.clim(0, 0.6)
plt.xlim(-300, 1000)
plt.title('ITPC')
plt.xlabel('Time (ms)')
plt.ylabel('Frequency (Hz)')
plt.show()

# %%
