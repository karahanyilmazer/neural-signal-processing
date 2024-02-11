# %%
#     COURSE: Solved problems in neural time series analysis
#    SECTION: Time-frequency analyses
#    TEACHER: Mike X Cohen, sincxpress.com
#      VIDEO: Narrowband filtering and the Hilbert transform

# !%matplotlib inline
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fft, ifft
from scipy.signal import filtfilt, firwin, freqz, hilbert

# Set figure settings
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import get_cmap, set_fig_dpi, set_style

set_fig_dpi(), set_style()
cmap = get_cmap('parula')

# %% Design FIR filter via firwin
srate = 1000
nyquist = srate / 2
freq_range = [10, 15]
order = round(20 * srate / freq_range[0])
order |= 1  # Ensure that order is odd

# Filter kernel
filt_kern = firwin(order + 1, freq_range, nyq=nyquist, pass_zero=False)

# Compute the power spectrum of the filter kernel
filt_pow = np.abs(np.fft.fft(filt_kern)) ** 2
# Compute the frequencies vector and remove negative frequencies
hz = np.linspace(0, srate / 2, int(np.floor(len(filt_kern) / 2)) + 1)
filt_pow = filt_pow[: len(hz)]

# %% Plot filter characteristics

plt.figure(figsize=(15, 5))
plt.subplot(131)
plt.plot(filt_kern)
plt.xlabel('Time Points')
plt.title('Filter Kernel (firwin)')

plt.subplot(132)
plt.plot(hz, filt_pow, 'ks-', markerfacecolor='w', markersize=8)
plt.plot(
    [0, freq_range[0], freq_range[0], freq_range[1], freq_range[1], nyquist],
    [0, 0, 1, 1, 0, 0],
    'ro-',
    markerfacecolor='w',
    label='Actual',
)
plt.plot([freq_range[0], freq_range[0]], plt.gca().get_ylim(), 'k:', label='Ideal')
plt.xlim(0, freq_range[0] * 4)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Filter gain')
plt.legend()
plt.title('Frequency Response of Filter (firwin)')

plt.subplot(133)
plt.plot(hz, 10 * np.log10(filt_pow), 'ks-', markersize=10, markerfacecolor='w')
plt.plot([freq_range[0], freq_range[0]], plt.gca().get_ylim(), 'k:')
plt.xlim(0, freq_range[0] * 4)
plt.ylim(-80, 2)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Filter Gain (dB)')
plt.title('Frequency Response of Filter (dB)')

plt.tight_layout()
plt.show()

# %% Apply the filter to the data
pnts = 10000
orig_data = np.random.randn(pnts)
time_vec = np.arange(pnts) / srate

filt_data = filtfilt(filt_kern, 1, orig_data)

# %% Hilbert transform

hil_data = hilbert(filt_data)

plt.figure(figsize=(10, 8))
plt.subplot(311)
plt.plot(time_vec, filt_data, 'k', label='Original')
plt.plot(time_vec, np.real(hil_data), 'ro', label='Real')
plt.title('Filtered signal')
plt.legend()

plt.subplot(312)
plt.plot(time_vec, np.real(hil_data), label='Real')
plt.plot(time_vec, np.imag(hil_data), label='Imag')
plt.plot(time_vec, np.abs(hil_data), 'k', label='Envelope')
plt.title('Complex Representation')
plt.legend()

plt.subplot(313)
plt.plot(time_vec, np.abs(hil_data), 'k')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Band-Limited Amplitude Envelope')

plt.tight_layout()
plt.show()

# %% Done.
