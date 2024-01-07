# %%
#   COURSE: Neural signal processing and analysis: Zero to hero
#  SECTION: Time-frequency analysis
#  TEACHER: Mike X Cohen, sincxpress.com
#    VIDEO: Getting to know Morlet wavelets

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
# Parameters
srate = 1000  # In Hz
# Best practice is to have time=0 at the center of the wavelet
time = np.arange(-1, 1 + 1 / srate, 1 / srate)
freq = 2 * np.pi  # Frequency of wavelet, in Hz

# Create sine wave (actually cosine, just to make it nice and symmetric)
sine_wave = np.cos(2 * np.pi * freq * time)

# Create Gaussian window
fwhm = 0.5  # width of the Gaussian in seconds
gauss_win = np.exp((-4 * np.log(2) * time**2) / (fwhm**2))


# Now create Morlet wavelet
mw = sine_wave * gauss_win

plt.subplot(211)
plt.plot(time, sine_wave, 'r', lw=1, label='Sine Wave')
plt.plot(time, gauss_win, 'b', lw=1, label='Gaussian')
plt.plot(time, mw, 'k', label='Morlet Wavelet')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend(loc='lower right')
plt.title('Morlet Wavelet in the Time Domain')

# Morlet wavelet in the frequency domain

# Confirm that the shape of the power spectrum of a Morlet wavelet is Gaussian

n_pnts = len(time)

mwX = np.abs(np.fft.fft(mw) / n_pnts)
hz = np.linspace(0, srate, n_pnts)

plt.subplot(212)
plt.plot(hz, mwX[: len(hz)], 'k')
plt.title('Morlet Wavelet in the Frequency Domain')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.xlim(0.0, 30)

plt.tight_layout()
plt.show()

# Observations: - Notice that the amplitude spectrum is symmetric.
#               - Also notice the peak amplitude in the time vs. frequency domains.
#               - The Hz units are incorrect above Nyquist. This is just a
#                 convenience plotting trick.
#                 (The frequencies above Nyquist are the negative ones.)
#
# TO DO: Change the following parameters to observe the effects:
#        - freq -->
#        - fwhm --> increases time resolution, decreases frequency resolution
#        - time (start and end values) --> increases frequency resolution

# %%
