# %%
#   COURSE: Neural signal processing and analysis: Zero to hero
#  SESSION: Static spectral analyses
#  TEACHER: Mike X Cohen, sincxpress.com
#    VIDEO: Welch's method on resting-state EEG data

# !%matplotlib inline
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from scipy.signal import welch

# Set figure settings
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import set_fig_dpi, set_style

set_fig_dpi(), set_style(notebook=True)

# %%
# Load the data
mat_file = loadmat(os.path.join('..', 'data', 'EEGrestingState.mat'))
data = mat_file['eegdata'][0]
srate = mat_file['srate'][0][0]
N = len(data)

# Time vector
time_vec = np.arange(0, N) / srate

# Plot the data
plt.plot(time_vec, data)
plt.title('Raw EEG Data')
plt.xlabel('Time (s)')
plt.ylabel('Voltage ($\mu$V)')

# %% One big FFT (not Welch's method)

# "Static" FFT over entire period, for comparison with Welch
eeg_pow = np.abs(np.fft.fft(data) / N) ** 2
hz = np.linspace(0, srate / 2, int(np.floor(N / 2)) + 1)

# %%  "Manual" Welch's method

# Window length in seconds*srate
win_len = 1 * srate

# Number of points of overlap
n_overlap = round(srate / 2)

# NOTE about the variable n_overlap:
# This variable actually defines the number of data
# points to skip forwards. The true number of overlapping
# points is win_len-n_skip. Apologies for the confusion,
# and thanks to Eleonora De Filippi for catching that mistake.


# Window onset times
win_onsets = np.arange(1, N - win_len + 1, n_overlap)

# Different-length signal needs a different-length Hz vector
hz_w = np.linspace(0, srate / 2, int(np.floor(win_len / 2) + 1))

# Hann window
hann = 0.5 - np.cos(2 * np.pi * np.linspace(0, 1, win_len)) / 2

# Initialize the power matrix (windows x frequencies)
eeg_pow_w = np.zeros(len(hz_w))

# Loop over frequencies
for onset in win_onsets:
    # Get a chunk of data from this time window
    data_chunk = data[onset : onset + win_len]

    # Apply Hann taper to data
    data_chunk = data_chunk * hann

    # Compute its power
    tmp_pow = np.abs(np.fft.fft(data_chunk) / win_len) ** 2

    # Enter into matrix
    eeg_pow_w = eeg_pow_w + tmp_pow[: len(hz_w)]

# Divide by N
eeg_pow_w /= len(win_onsets)

# %% Welch's method from Scipy

# Create Hann window
win_size = 2 * srate  # 2-second window
# Number of FFT points (frequency resolution)
n_fft = srate * 100
# Hann window
hann = 0.5 - np.cos(2 * np.pi * np.linspace(0, 1, win_size)) / 2
f, Pxx = welch(
    data,
    srate,
    window=hann,
    nperseg=win_size,
    noverlap=np.round(win_size / 4),
    nfft=srate * 100,
)

# Convert the power spectral density to decibels
Pxx_dB = 10 * np.log10(Pxx)

# %% Plotting
plt.figure()
plt.subplot(211)
plt.plot(hz, eeg_pow[: len(hz)], label='Static FFT')
plt.plot(hz_w, eeg_pow_w / 10, label="Welch's method")
plt.xlim([0, 40])
plt.ylim([0, 0.6])
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power')
plt.legend()
plt.title("Using FFT and Welch's Method")

plt.subplot(212)
plt.plot(f, Pxx_dB)
plt.xlim([0, 40])
plt.ylim([-20, 15])
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power')

plt.tight_layout()
plt.show()

# %%
