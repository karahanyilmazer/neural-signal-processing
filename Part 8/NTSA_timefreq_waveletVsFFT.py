# %%
#     COURSE: Solved problems in neural time series analysis
#    SECTION: Time-frequency analyses
#    TEACHER: Mike X Cohen, sincxpress.com
#      VIDEO: Compare wavelet-derived spectrum and FFT

# !%matplotlib inline
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

# Set figure settings
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import get_cmap, set_fig_dpi, set_style

set_fig_dpi(), set_style()
cmap = get_cmap('parula')

# %%
# Simulation parameters
srate = 1001
pnts = srate * 4
time = np.arange(0, pnts) / srate
time = time - np.mean(time)

# Chirp parameters (in Hz)
min_freq = 5
max_freq = 17

# Exponential function, scaled to [0 1]
fm = np.exp(time)
fm = fm - np.min(fm)
fm = fm / np.max(fm)

# Scale to frequency ranges
fm = fm * (max_freq - min_freq) + min_freq

# Generate chirp
churp = np.sin(2 * np.pi * ((time + np.cumsum(fm)) / srate))

# Add a pure sine wave
# churp = churp + 0.4 * np.sin(2 * np.pi * 13 * time)

# Plotting
plt.figure(figsize=(10, 8))

# Frequency time series
plt.subplot(311)
plt.plot(time, fm)
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')

# Plot the chirp
plt.subplot(312)
plt.plot(time, churp)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (a.u.)')

# Power spectrum
plt.subplot(313)
plt.plot(np.linspace(0, srate, pnts), np.abs(np.fft.fft(churp)) ** 2)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power')
plt.xlim([0, max_freq * 2])
plt.title('Static FFT')

plt.tight_layout()
plt.show()

# %%
# Wavelet parameters
min_freq = 2  # Hz
max_freq = 30  # Hz
n_freqs = 40

# Vector of wavelet frequencies
freqs = np.logspace(np.log10(min_freq), np.log10(max_freq), n_freqs)

# Gaussian parameters
fwhms = np.logspace(np.log10(0.5), np.log10(0.1), n_freqs)

# Initialize wavelet matrix
cmw = np.zeros((n_freqs, pnts), dtype=complex)

for fi in range(n_freqs):
    # Create complex sine wave
    csw = np.exp(1j * 2 * np.pi * freqs[fi] * time)

    # Create Gaussian
    gauss = np.exp(-4 * np.log(2) * time**2 / fwhms[fi] ** 2)

    # Create complex Morlet wavelets
    cmw[fi, :] = csw * gauss

# Convolution
nConv = 2 * pnts - 1
half_wave = pnts // 2

# Initialize time-frequency matrix
tf = np.zeros((n_freqs, pnts))

# FFT of data (doesn't change over frequencies!)
dataX = np.fft.fft(churp, nConv)

for fi in range(n_freqs):
    # Wavelet spectrum
    waveX = np.fft.fft(cmw[fi, :], nConv)
    waveX = waveX / np.max(waveX)

    # Convolution
    convres = np.fft.ifft(dataX * waveX)
    convres = convres[half_wave - 1 : -half_wave]

    # Extract power
    tf[fi, :] = np.abs(convres) ** 2

# Plot time-frequency representation
plt.figure(figsize=(10, 6))
plt.contourf(time, freqs, tf, 40, cmap=cmap)
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.show()

# %% Comparison

fft_power = (2 * np.abs(np.fft.fft(churp) / pnts)) ** 2
hz = np.linspace(0, srate / 2, pnts // 2 + 1)

plt.figure()
plt.plot(hz, fft_power[: len(hz)], label='FFT')
plt.plot(freqs, np.mean(tf, axis=1), label='Wavelet')
plt.xlim(0, max_freq * 2)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power')
plt.legend()
plt.show()

# %%
