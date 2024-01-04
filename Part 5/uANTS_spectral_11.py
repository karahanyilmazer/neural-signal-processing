# %%
#   COURSE: Neural signal processing and analysis: Zero to hero
#  SESSION: Static spectral analyses
#  TEACHER: Mike X Cohen, sincxpress.com
#    VIDEO: Reconstruct a signal via inverse Fourier transform

# !%matplotlib inline
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

# Set figure settings
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import set_fig_dpi, set_style

set_fig_dpi(), set_style(notebook=True)

# %% Create the signal

# Define a sampling rate and time vector
srate = 1000
time = np.arange(-1, 1 + 1 / srate, 1 / srate)

# Frequencies
freqs = [3, 10, 5, 15, 35]


# Now loop through frequencies and create sine waves
signal = np.zeros(len(time))
for fi in range(len(freqs)):
    signal = signal + (fi + 1) * np.sin(2 * np.pi * time * freqs[fi])

plt.plot(time, signal)
plt.title('Ground Truth')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (a.u.)')
plt.show()

# %%
# Here you will invert the Fourier transform, by starting from Fourier coefficients and
# getting back into the time domain.

N = len(signal)  # length of sequence
fourierTime = np.arange(N) / N  # "time" used for sine waves

reconSignal = np.zeros(signal.shape)
fourierCoefs = np.fft.fft(signal) / N

# Loop over frequencies
for fi in range(N):
    # Create coefficient-modulated sine wave for this frequency
    # Note: this is a complex sine wave without the minus sine in the exponential.
    fourierSine = fourierCoefs[fi] * np.exp(1j * 2 * np.pi * fi * fourierTime)

    # Continue building up signal...
    reconSignal = reconSignal + fourierSine

# Note: in practice, the inverse Fourier transform should be done using:
# reconSignal = np.fft.ifft(fourierCoefs) * N

plt.figure()
plt.plot(time, reconSignal.real, label='reconstructed')
plt.plot(time, signal, 'o', markerfacecolor='none', label='original')
plt.legend()
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (a.u.)')
plt.title('IFFT Results')
plt.show()

# %%
