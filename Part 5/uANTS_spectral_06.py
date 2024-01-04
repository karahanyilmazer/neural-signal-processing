# %%
#   COURSE: Neural signal processing and analysis: Zero to hero
#  SESSION: Static spectral analyses
#  TEACHER: Mike X Cohen, sincxpress.com
#    VIDEO: The discrete-time Fourier transform
#     GOAL: Implement the Fourier transform in a loop, as described in lecture.

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
# Generate multi-component sine wave (exactly as before)

# Define a sampling rate
srate = 1000

# Some parameters
freqs = [3, 10, 5, 15, 35]
amps = [5, 15, 10, 5, 7]

# Define time...
time = np.arange(-1, 1, 1 / srate)

# Loop through frequencies and create sine waves
signal = np.zeros(len(time))
for freq, amp in zip(freqs, amps):
    signal = signal + amp * np.sin(2 * np.pi * time * freq)

# %% The Fourier transform in a loop
nyquist = int(srate / 2)  # Nyquist freq.: the highest frequency you can measure in data
N = len(signal)  # Length of sequence
fourier_time = np.arange(N) / N  # "Time" used for sine waves

# Initialize Fourier output matrix
fourier_coefs = np.zeros(signal.shape, dtype=np.complex64)

# These are the actual frequencies in Hz that will be returned by the Fourier transform.
# The number of unique frequencies we can measure is exactly 1/2 of the number of data
# points in the time series (plus DC).
frequencies = np.linspace(0, nyquist, int(np.floor(N / 2) + 1))

# Loop over frequencies
for fi in range(N):
    # Create complex-valued sine wave for this frequency
    fourier_sine = np.exp(1j * 2 * np.pi * fi * fourier_time)

    # Compute dot product between sine wave and signal (created in the previous cell)
    fourier_coefs[fi] = np.dot(signal, fourier_sine)

# Scale Fourier coefficients to original scale
# --> Otherwise FT gets larger as the signal gets longer
fourier_coefs = fourier_coefs / N

fig = plt.figure()
gs = fig.add_gridspec(2, 2)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, 0:2])

ax1.plot(np.exp(-2 * np.pi * 1j * (10) * fourier_time).real)
ax1.set_title('One Sine Wave from the FT (Real Part)')
ax1.set_xlabel('Time (a.u.)')
ax1.set_ylabel('Amplitude (a.u.)')

ax2.plot(signal)
ax2.set_title('Signal')
ax1.set_xlabel('Time (a.u.)')

ax3.stem(frequencies, np.abs(fourier_coefs[: len(frequencies)] * 2))
ax3.set_xlim(0, 40)
ax3.set_xlabel('Frequency (Hz)')
ax3.set_ylabel('Power ($\mu$V)')
ax3.set_title('Power Spectrum Derived from Discrete Fourier Transform')
plt.tight_layout()
plt.show()

# %% The fast-Fourier transform (FFT)

# The "slow" FT is important to understand and see implemented, but in practice you
# should always use the FFT. In this code you will see that they produce the same results.

# Compute fourier transform and scale
fft_coefs = np.fft.fft(signal) / N

plt.figure()
plt.stem(frequencies, np.abs(fft_coefs[: len(frequencies)] * 2))
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power ($\mu$V)')
plt.title('Power Spectrum Derived from Fast Fourier Transform')
plt.xlim(0, 40)
plt.show()

# %%
