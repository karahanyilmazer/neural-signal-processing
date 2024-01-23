# %%
#   COURSE: Neural signal processing and analysis: Zero to hero
#  SECTION: Time-frequency analysis
#  TEACHER: Mike X Cohen, sincxpress.com
#    VIDEO: A full time-frequency power plot!

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
# %% Take a deep breath: You're about to make your first time-frequency power plot!

# Now you will observe convolution with real data
mat = loadmat(os.path.join('..', 'data', 'v1_laminar.mat'))
# Get the time points
time_vec = mat['timevec'][0]
# Get the sampling frequency
srate = mat['srate'][0][0]

# Extract all the trials from a single channel
data = mat['csd'][5, :, :]
data_shape = data.shape

# Reshape the data to be 1D --> form a "super-trial"
dataR = data.T.reshape(-1)

# Frequency parameters
min_freq = 5  # in Hz
max_freq = 90  # in Hz
num_freq = 30  # in count

freqs = np.linspace(min_freq, max_freq, num_freq)

time = np.arange(0, 2 * srate) / srate
time = time - np.mean(time)

n_data = len(dataR)
n_kern = len(time)
n_conv = n_data + n_kern - 1
half_k = int(np.floor(n_kern / 2))

dataX = np.fft.fft(dataR, n_conv)

# Initialize TF matrix
tf = np.zeros((num_freq, len(time_vec)))

for fi in range(num_freq):
    # Create wavelet
    cmw = np.exp(1j * 2 * np.pi * freqs[fi] * time) * np.exp(
        -4 * np.log(2) * time**2 / (0.3**2)
    )

    cmwX = np.fft.fft(cmw, n_conv)
    cmwX = cmwX / np.max(cmwX)

    # The rest of convolution
    analytic_signal = np.fft.ifft(dataX * cmwX)
    analytic_signal = analytic_signal[half_k : -half_k + 1]
    analytic_signal = analytic_signal.reshape(data.shape, order='F')

    # Extract power
    analytic_signal_pow = np.abs(analytic_signal) ** 2

    # Average over trials and put in matrix
    tf[fi, :] = np.mean(analytic_signal_pow, axis=1)

# %% Plotting
plt.contourf(time_vec, freqs, tf, 40)

plt.clim(0, 10000)
plt.xlim(-0.1, 1.4)

plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')

plt.show()

# %%
