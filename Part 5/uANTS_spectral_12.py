# %%
#   COURSE: Neural signal processing and analysis: Zero to hero
#  SESSION: Static spectral analyses
#  TEACHER: Mike X Cohen, sincxpress.com
#    VIDEO: Frequency resolution and zero-padding

# !%matplotlib inline
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

# Set figure settings
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import set_fig_dpi, set_style

set_fig_dpi(), set_style(notebook=True)
# %%
# We start by investigating the difference between sampling rate and
# number of time points for Fourier frequencies

# Temporal parameters
srates = [100, 100, 1000]
time_dur = [1, 10, 1]

# Define parameters
freq = 5  # Frequency in Hz
colors = ['k', 'm', 'b']  # Colors
symbols = ['o', 'p', '.']  # Symbols

fig, axs = plt.subplots(2, 1, figsize=(10, 8))

legendText = []
for parami in range(len(colors)):
    # Define sampling rate in this round
    srate = srates[parami]

    # Define time
    time = np.arange(-1, time_dur[parami] + 1 / srate, 1 / srate)

    # Create signal (Morlet wavelet)
    signal = np.cos(2 * np.pi * freq * time) * np.exp(-(time**2) / 0.05)

    # Compute FFT and normalize
    signalX = np.fft.fft(signal) / len(signal)
    signalX /= max(signalX)

    # Define vector of frequencies in Hz
    hz = np.linspace(0, srate / 2, int(np.floor(len(signal) / 2) + 1))

    # Plot time-domain signal
    axs[0].plot(time, signal, colors[parami] + symbols[parami] + '-', markersize=10)
    axs[0].set_xlim([-1, 1])
    axs[0].set_xlabel('Time (s)')
    axs[0].set_ylabel('Amplitude')
    axs[0].set_title('Time domain')

    # Plot frequency-domain signal
    axs[1].plot(
        hz,
        np.abs(signalX[: len(hz)]),
        colors[parami] + symbols[parami] + '-',
        markersize=10,
        label=f'srate={srates[parami]}, N={time_dur[parami]+1}s',
    )
    axs[1].set_xlabel('Frequency (Hz)')
    axs[1].set_ylabel('Amplitude')
    axs[1].set_title('Frequency domain')

axs[1].legend()

plt.tight_layout()
plt.show()

# %% Zero-padding, spectral resolution, and sinc interpolation

# Explore the effects of zero-padding.
# Note: I created the numbers here to show rather strong effects of zero-padding.
# You can try with other numbers the effects will be more mild.

signal = [1, 0, 1, 2, -3, 1, 2, 0, -3, 0, -1, 2, 1, -1]

# Compute the FFT of the same signal with different DC offsets
# No zero-padding
signalX1 = np.fft.fft(signal, len(signal)) / len(signal)
# Zero-pad to 10 + length of signal
signalX2 = np.fft.fft(signal, len(signal) + 10) / len(signal)
# Zero-pad to 100 + length of signal
signalX3 = np.fft.fft(signal, len(signal) + 100) / len(signal)

# Define frequencies vector
frex1 = np.linspace(0, 0.5, int(np.floor(len(signalX1) / 2)) + 1)
frex2 = np.linspace(0, 0.5, int(np.floor(len(signalX2) / 2)) + 1)
frex3 = np.linspace(0, 0.5, int(np.floor(len(signalX3) / 2)) + 1)


# Plot signals in the time domain
plt.subplot(211)
plt.plot(np.fft.ifft(signalX1) * len(signal), 'bo-')
plt.plot(np.fft.ifft(signalX2) * len(signal), 'rd-')
plt.plot(np.fft.ifft(signalX3) * len(signal), 'k*-')
plt.xlabel('Time points (arb. units)')
plt.ylabel('Amplitude')

# Plot signals in the frequency domain
plt.subplot(212)
plt.plot(frex1, 2 * np.abs(signalX1[: len(frex1)]), 'bo-')
plt.plot(frex2, 2 * np.abs(signalX2[: len(frex2)]), 'rd-')
plt.plot(frex3, 2 * np.abs(signalX3[: len(frex3)]), 'k*-')

plt.xlabel('Normalized frequency units')
plt.ylabel('Amplitude')
plt.legend(['"Native" N', 'N+10', 'N+100'])
plt.tight_layout()
plt.show()

# %%
