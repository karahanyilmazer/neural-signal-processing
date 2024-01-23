# %%
#   COURSE: Neural signal processing and analysis: Zero to hero
#  SECTION: Time-frequency analysis
#  TEACHER: Mike X Cohen, sincxpress.com
#    VIDEO: Complex Morlet wavelets

# !%matplotlib inline
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

# Set figure settings
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import set_fig_dpi, set_style

set_fig_dpi(), set_style()
# %%
# Setup parameters
srate = 1000  # In hz
# Best practice is to have time=0 at the center of the wavelet
time = np.arange(-1, 1 + 1 / srate, 1 / srate)
freq = 2 * np.pi  # Frequency of wavelet, in Hz

# Create sine wave
sine_wave = np.exp(1j * 2 * np.pi * freq * time)

# Create Gaussian window
fwhm = 0.5  # Width of the Gaussian in seconds
gauss_win = np.exp(-4 * np.log(2) * time**2 / fwhm**2)

# Now create Morlet wavelet
cmw = sine_wave * gauss_win

# %%
plt.figure()
plt.subplot(211)
plt.plot(time, cmw.real, 'b', label='Real Part')
plt.plot(time, cmw.imag, 'r--', label='Imaginary Part')
plt.title('Complex Morlet Wavelet in the Time Domain')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend()

# Complex Morlet wavelet in the frequency domain

n_pnts = len(time)

mwX = np.abs(np.fft.fft(cmw) / n_pnts)
hz = np.linspace(0, srate, n_pnts)

plt.subplot(212)
plt.plot(hz, mwX, 'k')
plt.title('Complex Morlet Wavelet in the Frequency Domain')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')

plt.tight_layout()
plt.show()

# Q: What happened to the spectrum? Is it still symmetric?
# A: No there are only the positive frequencies.
#    But it doesn't always have to be like that.

# %%
