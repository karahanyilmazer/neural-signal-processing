# %%
#     COURSE: Solved problems in neural time series analysis
#    SECTION: Time-frequency analyses
#    TEACHER: Mike X Cohen, sincxpress.com
#      VIDEO: Visualize time-frequency power from all channels

# !%matplotlib inline
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fft, ifft
from scipy.io import loadmat
from scipy.signal import filtfilt, firwin, hilbert, welch

# Set figure settings
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import set_fig_dpi, set_style

set_fig_dpi(), set_style()

# %%
# Simulation details
srate = 1000
time = np.arange(0, 4, 1 / srate)
pnts = len(time)
noise_amplitude = 0.1  # Noise amplitude --> if non-zero, smoothing must be applied

# Create signal (multipolar chirp)
k = 10  # Poles for frequencies
freq_mod = 20 * np.interp(np.linspace(0, k, pnts), np.arange(0, k), np.random.rand(k))
signal = np.sin(2 * np.pi * (time + np.cumsum(freq_mod) / srate))
signal += np.random.randn(len(signal)) * noise_amplitude

# Compute instantaneous frequency
signal_hilbert = hilbert(signal)
angles = np.angle(signal_hilbert)
inst_freq = np.diff(np.unwrap(angles)) / (2 * np.pi) * srate

# Static power spectrum
hz = np.linspace(0, srate / 2, int(np.floor(pnts / 2)) + 1)
amp = 2 * np.abs(np.fft.fft(signal) / pnts)
amp = amp[: len(hz)] ** 2

# Mean smoothing kernel
k = 13
kernel = np.ones(k) / k
nfft = len(inst_freq) + k - 1

# Apply mean smoothing
inst_freq_fft = fft(inst_freq, nfft)
kernel_fft = fft(kernel, nfft)
inst_freq_smooth = ifft(inst_freq_fft * kernel_fft)
inst_freq_smooth = inst_freq_smooth[int(np.ceil(k / 2)) - 1 : -int(np.ceil(k / 2))]


# Plotting
plt.figure()

# Time domain signal
plt.subplot(311)
plt.plot(time, signal, 'k')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Time Domain')

# Frequency domain
plt.subplot(312)
plt.plot(hz, amp, 'ks-', markerfacecolor='w')
plt.xlim([0, np.max(freq_mod) * 1.2])
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power')
plt.title('Frequency Domain')

# Instantaneous frequency
plt.subplot(313)
# plt.plot(time[:-1], inst_freq, 'k', label='Estimated')  # Non-smoothened
plt.plot(time[1:-1], inst_freq_smooth, 'k', label='Estimated')  # Smoothened
plt.plot(time, freq_mod, 'r', label='Ground Truth')
plt.ylim([0, np.max(freq_mod) * 1.2])
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.title('Instantaneous Frequency')
plt.legend()

plt.tight_layout()
plt.show()

# %% Now for real data
# Load the data
mat_file = loadmat(os.path.join('..', 'data', 'EEGrestingState.mat'))
data = mat_file['eegdata'][0]
srate = mat_file['srate'][0][0]
n_samples = len(data)

# Get the time vector
time = np.arange(0, n_samples) / srate

frequencies, psd = welch(
    data, fs=srate, nperseg=srate, noverlap=srate // 2, nfft=srate * 2
)
psd = 10 * np.log10(psd)  # Convert to dB

plt.figure()
plt.plot(frequencies, psd)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power')
plt.title('Welch Power Spectral Density Estimate')
plt.xlim(0, 40)
plt.ylim(-15, 15)
plt.show()

# %% Narrowband filter around 10 Hz (alpha band)
# Filter parameters
nyquist = srate / 2
freq_range = np.array([8, 12])
order = round(20 * srate / freq_range[0])
order |= 1  # Ensure it is odd

# Filter kernel
filt_kern = firwin(order + 1, freq_range, nyq=nyquist, pass_zero=False)

# Compute the power spectrum of the filter kernel
filt_pow = np.abs(np.fft.fft(filt_kern)) ** 2
# Compute the frequencies vector and remove negative frequencies
hz = np.linspace(0, srate / 2, int(np.floor(len(filt_kern) / 2)) + 1)
filt_pow = filt_pow[: len(hz)]

# Visualize the filter kernel
plt.figure()
plt.subplot(121)
plt.plot(filt_kern)
plt.xlabel('Time Points')
plt.title('Filter Kernel (firwin)')

# Plot amplitude spectrum of the filter kernel
plt.subplot(122)
plt.plot(hz, filt_pow, 'ks-', markerfacecolor='w', label='actual')
plt.plot(
    [0, freq_range[0], freq_range[0], freq_range[1], freq_range[1], nyquist],
    [0, 0, 1, 1, 0, 0],
    'ro-',
    markerfacecolor='w',
    label='ideal',
)
plt.plot([freq_range[0], freq_range[0]], plt.gca().get_ylim(), 'k:')
plt.xlim(0, freq_range[0] * 4)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Filter gain')
plt.legend()
plt.title('Frequency Response of Filter (firwin)')

plt.tight_layout()
plt.show()

# %%
# Apply the filter to the data
filt_data = filtfilt(filt_kern, 1, data.astype(float))

# Plot the data for comparison
plt.figure()
plt.plot(time, data, label='Original')
plt.plot(time, filt_data, label='Filtered')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend()
plt.show()

# %% Compute instantaneous frequency
angles = np.angle(hilbert(filt_data))
inst_alpha = np.diff(np.unwrap(angles)) / (2 * np.pi / srate)

plt.figure()
plt.plot(time[:-1], inst_alpha, 'ks-', markerfacecolor='w')
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.show()

# %% Apply median filter to supra-threshold points

# Convert to z-score and show histogram
inst_z = (inst_alpha - np.mean(inst_alpha)) / np.std(inst_alpha)

plt.figure()
plt.hist(inst_z, 200)
plt.xlabel('I.F. (z)'), plt.ylabel('Count')
plt.title('Distribution of z-normalized Instantaneous Frequency')
plt.show()

# %% Cut off large deviations using median filter

# Identify supra-threshold data points
# 2 standard deviations corresponds to 95% confidence interval
to_filter = np.where(np.abs(inst_z) > 2)[0]

# Now for median filter
inst_alpha_filt = inst_alpha.copy()
k = round(1000 * 50 / srate)  # Median kernel size is 2k+1, where k is time in ms
for i in to_filter:
    indices = np.arange(max(0, i - k), min(len(inst_alpha), i + k + 1))
    inst_alpha_filt[i] = np.median(inst_alpha[indices])

plt.figure()
plt.plot(time[:-1], inst_alpha, 'k', label='Original')
plt.plot(time[:-1], inst_alpha_filt, 'r--', label='Filtered')
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.legend()
plt.ylim([5, 15])
plt.show()

# %%
