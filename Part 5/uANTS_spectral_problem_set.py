# %%
#   COURSE: Neural signal processing and analysis: Zero to hero
#  SESSION: Problem set: Spectral analyses of real and simulated data
#  TEACHER: Mike X Cohen, sincxpress.com

# !%matplotlib inline
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

# Set plt.figure settings
sys.path.insert(0, os.path.np.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import set_fig_dpi, set_style

set_fig_dpi(), set_style(notebook=True)
# %% Task 1
# Generate 10 seconds of data at 1 kHz, comprising 4 sine waves with different
# frequencies (between 1 and 30 Hz) and different amplitudes.
# Plot the individual sine waves, each in its own plot. In separate plt.subplots,
# plot the summed sine waves with (1) a little bit of noise and (2) a lot of noise.

srate = 1000
freqs = [1, 7, 12, 30]
amps = [3, 15, 10, 5]
phases = [0, np.pi / 7, np.pi / 8, np.pi / 2]
time = np.arange(-1, 1 + 1 / srate, 1 / srate)
N = len(time)

# Create sine waves, first initialize to correct size
sine_waves = np.zeros((len(freqs), N))

for i in range(len(freqs)):
    sine_waves[i, :] = amps[i] * np.sin(2 * np.pi * time * freqs[i] + phases[i])

signal = np.sum(sine_waves, axis=0)

little_noise = np.random.randn(*time.shape) * 10
large_noise = np.random.randn(*time.shape) * 50

# Plot constituent sine waves (without noise)
plt.figure()
for i in range(len(freqs)):
    plt.subplot(len(freqs), 1, i + 1)
    plt.plot(time, sine_waves[i, :])
    plt.title(f'Sine Wave Component with Frequency {freqs[i]} Hz')
    plt.ylabel('Amplitude (a.u.)')

plt.xlabel('Time (s)')
plt.tight_layout()
plt.show()

# Plot summed sine waves
plt.figure()
plt.subplot(211)
plt.plot(time, signal + little_noise)
plt.title('Time Series with Little Noise')

plt.subplot(212)
plt.plot(time, signal + large_noise)
plt.title('Time Series with Large Noise')

plt.tight_layout()
plt.show()

# %% Task 2
# Compute the power spectrum of the simulated time series (use FFT) and plot the results
# separately for a little noise and a lot of noise. Show frequencies 0 to 35 Hz.
# How well are the frequencies reconstructed, and does this depend on noise?

hz = np.linspace(0, srate / 2, int(np.floor(N / 2) + 1))
signalX_little = np.fft.fft(signal + little_noise).real[: len(hz)]
signalX_little = np.abs(signalX_little) / N
signalX_little[1:-1] *= 2

signalX_large = np.fft.fft(signal + large_noise).real[: len(hz)]
signalX_large = np.abs(signalX_large) / N
signalX_large[1:-1] *= 2

plt.figure()
plt.subplot(211)
plt.plot(hz, np.abs(signalX_little))
plt.title('FFT with Little Noise')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude (a.u.)')
plt.xlim(0, 35)

plt.subplot(212)
plt.plot(hz, np.abs(signalX_large))
plt.title('FFT with Large Noise')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude (a.u.)')
plt.xlim(0, 35)

plt.tight_layout()
plt.show()

# %% Task 3
# Compute the power spectrum of data from electrode 7 in the laminar V1 data.
# First, compute the power spectrum separately for each trial and then average the power
# results together. Next, average the trials together and then compute the power spectrum.
# Do the results look similar or different, and are you surprised? Why might they look
# similar or different?

# Load the LFP data
mat = loadmat(os.path.join('..', 'data', 'V1_Laminar.mat'))
time = mat['timevec'][0]
srate = mat['srate'][0][0]
data = mat['csd']

# Pick which channel to use
chan = 6

# FFT of all trials individually (note that you can do it in one line!)
pow_spect_sep = np.fft.fft(data[chan, :, :], axis=0)
pow_spect_sep = np.abs(pow_spect_sep) / len(time)
pow_spect_sep[1:-1] *= 2
pow_spect_sep = pow_spect_sep**2

# Then average the single-trial spectra together
# (Average over trials, not over frequencies)
pow_spect_sep = np.mean(pow_spect_sep, axis=1)

# Now average first, then take the FFT of the trial average
pow_spect_avg = np.mean(data[chan, :, :], axis=1)
pow_spect_avg = np.fft.fft(pow_spect_avg, axis=0)
pow_spect_avg = np.abs(pow_spect_avg) / len(time)
pow_spect_avg[1:-1] *= 2
pow_spect_avg = pow_spect_avg**2

# Frequencies in Hz
hz = np.linspace(0, srate / 2, int(np.floor(len(time) / 2) + 1))

# Plotting
plt.figure()
plt.suptitle(f'Results from Electrode {chan+1}')
plt.subplot(211)
plt.plot(hz, pow_spect_sep[: len(hz)])
plt.title('FFTs of Each Trial Averaged Together')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.xlim(0, 100)

plt.subplot(212)
plt.plot(hz, pow_spect_avg[: len(hz)])
plt.title('FFT of Average of All Trials')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.xlim(0, 100)

plt.tight_layout()
plt.show()

# %% Task 4
# Do the same as above but for electrode 1.
# How do these results compare to the results from channel 7, and does this depend on
# whether you average first in the time-domain or average the individual power spectra?
# ANATOMICAL NOTE: channel 7 is around L4 channel 1 is in the hippocampus.

# Pick which channel to use
chan = 0

# FFT of all trials individually (note that you can do it in one line!)
pow_spect_sep = np.fft.fft(data[chan, :, :], axis=0)
pow_spect_sep = np.abs(pow_spect_sep) / len(time)
pow_spect_sep[1:-1] *= 2
pow_spect_sep = pow_spect_sep**2

# Then average the single-trial spectra together
# (Average over trials, not over frequencies)
pow_spect_sep = np.mean(pow_spect_sep, axis=1)

# Now average first, then take the FFT of the trial average
pow_spect_avg = np.mean(data[chan, :, :], axis=1)
pow_spect_avg = np.fft.fft(pow_spect_avg, axis=0)
pow_spect_avg = np.abs(pow_spect_avg) / len(time)
pow_spect_avg[1:-1] *= 2
pow_spect_avg = pow_spect_avg**2

# Frequencies in Hz
hz = np.linspace(0, srate / 2, int(np.floor(len(time) / 2) + 1))

# Plotting
plt.figure()
plt.suptitle(f'Results from Electrode {chan+1}')
plt.subplot(211)
plt.plot(hz, pow_spect_sep[: len(hz)])
plt.title('FFTs of Each Trial Averaged Together')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.xlim(0, 100)

plt.subplot(212)
plt.plot(hz, pow_spect_avg[: len(hz)])
plt.title('FFT of Average of All Trials')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.xlim(0, 100)

plt.tight_layout()
plt.show()

# %% Task 5
# Fourier transform from scratch!
# Hey, wouldn't it be fun to program the discrete-time Fourier transform from scratch!
# Generate a 20-element vector of random numbers.
# Use the hints below to help you write the Fourier transform.
# Next, use the fft function on the same data to verify that your FT was accurate.

N = 20  # Length of sequence
signal = np.random.randn(N)  # Data
f_time = np.arange(N) / N  # "Time" used in Fourier transform

# Initialize Fourier output matrix
four_coefs = np.zeros_like(signal, dtype=complex)

# Loop over frequencies
for fi in range(N):
    # Create sine wave for this frequency
    four_sine = np.exp(1j * 2 * np.pi * fi * f_time)

    # Compute dot product as sum of point-wise elements
    four_coefs[fi] = np.dot(signal, four_sine)

# Divide by N to scale coefficients properly
four_coefs /= N

plt.figure()
plt.subplot(211)
plt.plot(signal)
plt.title('Signal')
plt.xlabel('Time (a.u.)')
plt.ylabel('Amplitude')

plt.subplot(212)
plt.plot(np.abs(four_coefs) * 2, '*-', label='Manual FT')
plt.title('FFT')
plt.xlabel('Frequency (a.u.)')
plt.ylabel('Amplitude')

# For comparison, use the fft function on the same data
four_coefs_fft = np.fft.fft(signal) / N

# Plot the results on top. Do they look similar? (Should be identical!)
plt.plot(np.abs(four_coefs_fft) * 2, 'ro', markerfacecolor='none', label='FFT')
plt.legend()

plt.tight_layout()
plt.show()

# %% Task 6
# Zero-padding and interpolation
# Compute the power spectrum of channel 7 from the V1 dataset. Take the
# power spectrum of each trial and then average the power spectra together.
# But don't use a loop over trials! And use only the data from 0-0.5 sec.
# What is the frequency resolution?
# Repeat this procedure, but zero-pad the data to increase frequency
# resolution. Try some different zero-padding numbers. At what multiple
# of the native nfft does the increased frequency resolution have no
# appreciable visible effect on the results?

# Time indices for 0 to 0.5 seconds
t_idx = [np.argmin(np.abs(time - 0)), np.argmin(np.abs(time - 0.5))]

# Set nfft to be multiples of the length of the data
nfft1 = 1 * (t_idx[1] - t_idx[0] + 1)
nfft2 = 2 * (t_idx[1] - t_idx[0] + 1)
nfft3 = 10 * (t_idx[1] - t_idx[0] + 1)

win = np.arange(t_idx[0], t_idx[1] + 1)

# First NFFT
hz = np.linspace(0, srate / 2, int(np.floor(nfft1 / 2)) + 1)
pow_spect = np.fft.fft(data[chan, win, :], n=nfft1, axis=0) ** 2
pow_spect = np.mean(2 * np.abs(pow_spect) / len(time), axis=1)

plt.figure()
plt.plot(hz, pow_spect[: len(hz)], '-o', markerfacecolor='none')
plt.title(f'Frequency resolution is {np.mean(np.diff(hz))} Hz')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power')
plt.xlim(0, 120)
plt.show()

# Second NFFT
hz = np.linspace(0, srate / 2, int(np.floor(nfft2 / 2)) + 1)
pow_spect = np.fft.fft(data[chan, win, :], n=nfft2, axis=0) ** 2
pow_spect = np.mean(2 * np.abs(pow_spect) / len(time), axis=1)

plt.figure()
plt.plot(hz, pow_spect[: len(hz)], '-o', markerfacecolor='none')
plt.title(f'Frequency resolution is {np.round(np.mean(np.diff(hz)), 4)} Hz')
plt.ylabel('Power')
plt.xlim(0, 120)
plt.show()

# Third NFFT
hz = np.linspace(0, srate / 2, int(np.floor(nfft3 / 2)) + 1)
pow_spect = np.fft.fft(data[chan, win, :], n=nfft3, axis=0) ** 2
pow_spect = np.mean(2 * np.abs(pow_spect) / len(time), axis=1)

plt.figure()
plt.plot(hz, pow_spect[: len(hz)], '-o', markerfacecolor='none')
plt.title(f'Frequency resolution is {np.round(np.mean(np.diff(hz)), 4)} Hz')
plt.ylabel('Power')
plt.xlim(0, 120)
plt.show()

# %% Task 7
# Poor man's filter via frequency-domain manipulations.
# The goal of this exercise is to see how a basic frequency-domain filter
# works: Take the FFT of a signal, zero-out some Fourier coefficients,
# take take the IFFT.
# Here you will do this by generating a 1/f noise signal and adding 50 Hz
# line noise.

# Generate 1/f noise with 50 Hz line noise.
srate = 1234
n_pnts = srate * 3
time = np.arange(n_pnts) / srate

# The key parameter of pink noise is the exponential decay (ed)
ed = 50  # Try different values!
amp_spec = np.random.rand(n_pnts) * np.exp(-np.arange(n_pnts) / ed)
fc = amp_spec * np.exp(1j * 2 * np.pi * np.random.rand(*amp_spec.shape))

# Construct the signal
signal = np.fft.ifft(fc).real * n_pnts

# Now add 50 Hz line noise
signal += np.sin(2 * np.pi * 50 * time)

# Compute its spectrum
hz = np.linspace(0, srate / 2, int(np.floor(n_pnts / 2)) + 1)
signalX = np.fft.fft(signal)

# Plot the signal and its power spectrum
plt.figure()
plt.suptitle('Signal with 50 Hz Noise')
plt.subplot(211)
plt.plot(time, signal)
plt.title('Time Domain')
plt.xlabel('Time (s)')
plt.ylabel('Activity')

plt.subplot(212)
plt.plot(hz, 2 * np.abs(signalX[: len(hz)]) / n_pnts, '-o')
plt.title('Frequency Domain')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.xlim(0, 80)

plt.tight_layout()
plt.show()

# %% Zero-out the 50 Hz component

# Find the index into the frequencies vector at 50 hz
hz_50_idx = np.argmin(np.abs(hz - 50))

# Create a copy of the Fourier coefficients
signalXf = signalX.copy()  # f for filter

# Zero out the 50 Hz component
signalXf[hz_50_idx] = 0

# Take the IFFT
signalf = np.fft.ifft(signalXf).real

# Take FFT of filtered signal
signalXf = np.fft.fft(signalf)

# Plot on top of original signal
plt.figure()
plt.suptitle('Signal with 50 Hz (Half-Filtered)')
plt.subplot(211)
plt.plot(time, signal)
plt.plot(time, signalf.real)
plt.title('Time Domain')
plt.xlabel('Time (s)')
plt.ylabel('Activity')

plt.subplot(212)
plt.plot(hz, 2 * np.abs(signalX[: len(hz)]) / n_pnts, '-o')
plt.plot(hz, 2 * np.abs(signalXf[: len(hz)]) / n_pnts, '-o')
plt.title('Frequency Domain')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.xlim(0, 80)

plt.tight_layout()
plt.show()


# Q: Why do you need to take the real part of the ifft?
# A: Because the ifft returns a complex number, and we want to plot a real number.
# Q: Why doesn't this procedure get rid of the line noise?!?!?!?
# A: Because the line noise is in the negative frequencies, and we only zeroed-out
#    the positive frequencies.

# %% Now fix the problem

# Notice that the filter didn't work: It attenuated but did not eliminate
# the line noise. Why did this happen? Use plotting to confirm your
# hypothesis! Then fix the problem in this cell.

# Find the index into the frequencies vector at 50 hz
hz_50_idx = np.argmin(np.abs(hz - 50))

# Create a copy of the frequency vector
signalXff = signalX.copy()  # f for filter

# Zero out the 50 Hz component (hint: the negative frequencies)
signalXff[hz_50_idx] = 0
signalXff[-hz_50_idx] = 0

# Take the IFFT
signalff = np.fft.ifft(signalXff).real

# Take FFT of filtered signal
signalXff = np.fft.fft(signalff)

# Plot all three versions
plt.figure()
plt.suptitle('Signal with 50 Hz (Fully Filtered)')
plt.subplot(211)
plt.plot(time, signal, label='Original')
plt.plot(time, signalf, label='Half-Filtered')
plt.plot(time, signalff, label='Filtered')
plt.title('Time Domain')
plt.xlabel('Time (s)')
plt.ylabel('Activity')

plt.subplot(212)
plt.plot(hz, 2 * np.abs(signalX[: len(hz)]) / n_pnts, '-o', label='Original')
plt.plot(hz, 2 * np.abs(signalXf[: len(hz)]) / n_pnts, '-o', label='Half-Filtered')
plt.plot(hz, 2 * np.abs(signalXff[: len(hz)]) / n_pnts, '-o', label='Filtered')
plt.title('Frequency Domain')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.xlim(0, 80)
plt.legend()

plt.tight_layout()
plt.show()

# Zeroing out one component introduces edge artifacts. These artifacts are not that
# significant in this case as the neighboring components have small amplitudes.

# %% Task 8
# Exploring why the filter in #7 isn't a good filter

# Number of time points
N = 1000

# Generate a flat Fourier spectra
four_spect1 = np.ones(N)

# Copy it and zero-out some fraction
four_spect2 = four_spect1.copy()
four_spect2[round(N * 0.1) : round(N * 0.2)] = 0

# Create time-domain signals via IFFT of the spectra
signal1 = np.fft.ifft(four_spect1).real
signal2 = np.fft.ifft(four_spect2).real
time = np.linspace(0, 1, N)

# Plotting
plt.figure()
plt.subplot(211)
plt.plot(time, four_spect1, label='Flat Spectrum')
plt.plot(time, four_spect2, label='With Edges')
plt.xlabel('Frequency (a.u.)')
plt.title('Frequency domain')
plt.legend()

plt.subplot(212)
plt.plot(time, signal2, label='With Edges')
plt.plot(time, signal1, label='Flat Spectrum')
plt.ylim(np.array([-1, 1]) * 0.05)
plt.xlabel('Time (a.u.)')
plt.title('Time Domain')
plt.legend()

plt.tight_layout()
plt.show()

# Brickwall filter adds a lot of ripples in the time domain.

# %%
