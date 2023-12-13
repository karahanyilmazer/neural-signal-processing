# %%
#     COURSE: Solved challenges in neural time series analysis
#    SECTION: Simulating EEG data
#      VIDEO: Project 1: Channel-level EEG data
# Instructor: sincxpress.com
#       Goal: Simulate time series data that can be used to
#             test time-series analysis methods

# !%matplotlib inline
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mne import create_info
from mne.channels import make_standard_montage
from mne.viz import plot_topomap
from scipy.io import loadmat

# Set figure settings
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import get_cmap, set_fig_dpi, set_style

set_fig_dpi(), set_style()
cmap = get_cmap('parula')


# %%
def plot_EEG(times, data, ch_idx=0):
    # Convert s to ms
    times = times.copy() * 1000

    # Initialize the figure
    mosaic = [['ERP', 'ERP'], ['PSD', 'TF']]
    fig = plt.figure(constrained_layout=True)
    ax_dict = fig.subplot_mosaic(mosaic)

    # Frequencies in Hz for the PSD plot
    freqs_psd = np.linspace(0, srate, n_samples)

    # Frequencies in Hz (hard-coded to 2 to 30 in 40 steps)
    freqs = np.linspace(2, 30, 40)
    # Number of wavelet cycles (hard-coded to 3 to 10)
    waves = 2 * (np.linspace(3, 10, len(freqs)) / (2 * np.pi * freqs)) ** 2

    # Setup wavelet and convolution parameters
    wave_t = np.arange(-2, 2 + 1 / srate, 1 / srate)
    half_w = int(np.floor(len(wave_t) / 2) + 1)
    n_conv = n_samples * n_trials + len(wave_t) - 1

    if data.ndim == 3:
        # ERP
        ax_dict['ERP'].plot(times, data[:, ch_idx, :].T, color='grey', alpha=0.1)
        ax_dict['ERP'].plot(times, np.mean(data[:, ch_idx, :].T, axis=1), color='black')

        # PSD
        # Perform FFT along the columns (trials)
        pw = np.mean(
            (2 * np.abs(np.fft.fft(data[:, ch_idx, :].T, axis=0) / n_samples)) ** 2,
            axis=1,
        )

        # TF
        data_fft = np.fft.fft(data[:, ch_idx, :].T.reshape(1, -1, order='F'), n_conv)
    else:
        # ERP
        ax_dict['ERP'].plot(times, data[ch_idx, :], color='black')

        # PSD
        pw = (2 * np.abs(np.fft.fft(data[ch_idx, :], axis=0) / n_samples)) ** 2

        # TF
        data_fft = np.fft.fft(data[ch_idx, :].reshape(1, -1, order='F'), n_conv)

    # Initialize the time-frequency matrix
    tf_mat = np.zeros((len(freqs), n_samples))

    for i in range(len(freqs)):
        wave_x = np.fft.fft(
            np.exp(2 * 1j * np.pi * freqs[i] * wave_t)
            * np.exp(-(wave_t**2) / waves[i]),
            n_conv,
        )
        wave_x = wave_x / np.max(wave_x)

        amp_spec = np.fft.ifft(wave_x * data_fft)
        amp_spec = amp_spec[0][half_w - 1 : -half_w + 1].reshape(
            (n_samples, n_trials), order='F'
        )

        tf_mat[i, :] = np.mean(np.abs(amp_spec), axis=1)

    ax_dict['ERP'].set_title(f'ERP From Channel {ch_idx+1}')
    ax_dict['ERP'].set_xlabel('Time (ms)')
    ax_dict['ERP'].set_ylabel('Activity')
    ax_dict['ERP'].set_xlim(times.min(), times.max())

    ax_dict['PSD'].plot(freqs_psd, pw)
    ax_dict['PSD'].set_title(f'Static Power Spectrum')
    ax_dict['PSD'].set_xlabel('Frequency (Hz)')
    ax_dict['PSD'].set_ylabel('Power')
    ax_dict['PSD'].set_xlim(0, 40)

    ax_dict['TF'].contourf(times, freqs, tf_mat, 40, cmap=cmap)
    ax_dict['TF'].set_title(f'Time-Frequency Plot')
    ax_dict['TF'].set_xlabel('Time (ms)')
    ax_dict['TF'].set_ylabel('Frequency (Hz)')
    ax_dict['TF'].set_xlim(times.min(), times.max())

    plt.show()


n_chs = 23
n_trials = 30
n_samples = 1500
srate = 500  # Hz
times = np.arange(n_samples) / srate
sine_freq = 6.75  # Hz

# info = create_info(ch_df['label'].to_list(), srate, 'eeg')
# montage = make_standard_montage('standard_1020')
# info.set_montage(montage)

# %% 1) Pure phase-locked sine wave

# Initialize data array
data = np.zeros((n_trials, n_chs, n_samples))

# Loop over channels and trials
for ch in range(n_chs):
    for trial in range(n_trials):
        data[trial, ch, :] = np.sin(2 * np.pi * sine_freq * times)


# Plot an ERP from one channel
plot_EEG(times, data, ch_idx=9)

# %% 2) Non-phase-locked sine wave

# Initialize data array
data = np.zeros((n_trials, n_chs, n_samples))

# Loop over channels and trials
for ch in range(n_chs):
    for trial in range(n_trials):
        data[trial, ch, :] = np.sin(
            2 * np.pi * sine_freq * times + np.random.rand() * 2 * np.pi
        )

plot_EEG(times, data, ch_idx=9)

# %% 3) Multisine waves

# List of frequencies and corresponding amplitudes
freqs = [3, 5, 16]
amps = [2, 4, 5]

# Initialize data array
data = np.zeros((n_trials, n_chs, n_samples))

# Loop over channels and trials
for ch in range(n_chs):
    for trial in range(n_trials):
        for freq, amp in zip(freqs, amps):
            data[trial, ch, :] += amp * np.sin(2 * np.pi * freq * times)
            # data[trial, ch, :] += amp*np.sin(2*np.pi*freq*times + np.random.rand()*2*np.pi)

plot_EEG(times, data, ch_idx=9)

# Q: What can you change in the code above to make the EEG
#    activity non-phase-locked over trials?
# A: Add a random number to the phase of each sine wave
#
# Q: Which of the plots look different for phase-locked vs. non-phase-locked?
#    (Hint: plot them in different figures to facilitate comparison.)
#    Are you surprised about the differences?
# A: The ERP and TF plots look different. The PSD plot looks the same.

# %%
## 4) Nonstationary sine waves

# Initialize data array
data = np.zeros((n_trials, n_chs, n_samples))

# Loop over channels and trials
for ch in range(n_chs):
    for trial in range(n_trials):
        for freq, amp in zip(freqs, amps):
            # Create instantaneous frequencies via interpolated random numbers
            freq_mod = 20 * np.interp(
                np.linspace(1, 10, n_samples),
                np.linspace(1, 10, 10),
                np.random.rand(10),
            )
            signal = np.sin(2 * np.pi * ((times + np.cumsum(freq_mod)) / srate))
            data[trial, ch, :] += signal

plot_EEG(times, data, ch_idx=9)

# %%
## 5) Transient oscillations with Gaussian

peaktime = 1  # second (where the peak occurs)
width = 0.12
gauss = np.exp(-((times - peaktime) ** 2) / (2 * width**2))

sine_freq, amp = 7, 1

# Initialize data array
data = np.zeros((n_trials, n_chs, n_samples))

# Loop over channels and trials
for ch in range(n_chs):
    for trial in range(n_trials):
        data[trial, ch, :] = amp * np.cos(2 * np.pi * sine_freq * times) * gauss

plot_EEG(times, data, ch_idx=9)

# %%
## 6) Repeat 3) with white noise

# Noise amplitude
noise_amp = 5

# List of frequencies and corresponding amplitudes
freqs = [3, 5, 16]
amps = [2, 4, 5]

# Initialize data array
data = np.zeros((n_trials, n_chs, n_samples))

# Loop over channels and trials
for ch in range(n_chs):
    for trial in range(n_trials):
        for freq, amp in zip(freqs, amps):
            signal = amp * np.sin(2 * np.pi * freq * times)
            noise = noise_amp * np.random.randn(*signal.shape)
            data[trial, ch, :] += signal + noise

plot_EEG(times, data, ch_idx=9)

# %%
## 7) Transient oscillations with 1/f noise

# Noise amplitude
noise_amp = 0.3
signal_amp = 1

peaktime = 1  # second (where the peak occurs)
width = 0.12
gauss = np.exp(-((times - peaktime) ** 2) / (2 * width**2))

# Initialize data array
data = np.zeros((n_trials, n_chs, n_samples))

# Parameters for the pink noise
exp_decay = 50
half_pnts = int(np.floor(n_samples / 2) - 1)

# Loop over channels and trials
for ch in range(n_chs):
    for trial in range(n_trials):
        # Create pink noise
        signal = np.cos(2 * np.pi * sine_freq * times + 2 * np.pi * np.random.rand())

        # Pink noise amplitude spectrum
        amp_spec = np.random.rand(half_pnts) * np.exp(-np.arange(half_pnts) / exp_decay)
        amp_spec = np.concatenate(([amp_spec[0]], amp_spec, [0], amp_spec[::-1]))

        # Fourier coefficients
        fc = amp_spec * np.exp(1j * 2 * np.pi * np.random.rand(*amp_spec.shape))

        # Inverse Fourier transform to create the noise
        pink_noise = np.fft.ifft(fc).real * n_samples

        data[trial, ch, :] = signal_amp * signal * gauss + noise_amp * pink_noise

plot_EEG(times, data, ch_idx=9)

# %%
