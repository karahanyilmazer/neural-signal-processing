# %%
#     COURSE: Solved problems in neural time series analysis
#    SECTION: Time-frequency analyses
#    TEACHER: Mike X Cohen, sincxpress.com
#      VIDEO: Wavelet convolution of close frequencies

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
srate = 1000
n_pnts = srate * 3  # 3 seconds
time = np.arange(0, n_pnts) / srate - 1

# Sine wave parameters (in Hz)
freq_1 = 10
freq_2 = 13

# Gaussian parameter (in seconds)
fwhm = 0.3
p_time_1 = 1
p_time_2 = 1.1

# Generate signals
sig_1 = np.sin(2 * np.pi * freq_1 * time) * np.exp(
    -4 * np.log(2) * ((time - p_time_1) / fwhm) ** 2
)
sig_2 = np.sin(2 * np.pi * freq_2 * time) * np.exp(
    -4 * np.log(2) * ((time - p_time_2) / fwhm) ** 2
)

# Add a touch of noise
sig_1 += 0.1 * np.random.randn(len(sig_1))
sig_2 += 0.1 * np.random.randn(len(sig_2))

# Now for the signal itself
signal = sig_1 + sig_2

# Plot the signal components
plt.figure()
plt.subplot(211)
plt.plot(time, sig_1, 'b', label='Signal 1')
plt.plot(time, sig_2, 'r', label='Signal 2')
plt.xlabel('Time (s)')
plt.ylabel('Amp. (a.u.)')
plt.legend()

# And the final signal
plt.subplot(212)
plt.plot(time, signal, 'k')
plt.xlabel('Time (s)')
plt.ylabel('Amp. (a.u.)')

plt.tight_layout()
plt.show()

# %%
# Time-frequency parameters
min_freq = 2
max_freq = 20
n_freqs = 60

wave_time = np.arange(-2, 2 + 1 / srate, 1 / srate)

# Vector of wavelet frequencies
freqs = np.logspace(np.log10(min_freq), np.log10(max_freq), n_freqs)

# Gaussian parameter
# wave_fwhm = 0.5
wave_fwhm = 1.5
# wave_fwhm = 2

n_conv = n_pnts + len(wave_time) - 1
half_w = len(wave_time) // 2

# Initialize time-frequency matrix
tf = np.zeros((n_freqs, n_pnts))

# FFT of the signal (doesn't change over frequencies!)
signalX = np.fft.fft(signal, n_conv)

for fi in range(n_freqs):
    # Create complex Morlet wavelet parts
    csw = np.exp(1j * 2 * np.pi * freqs[fi] * wave_time)
    gauss = np.exp(-4 * np.log(2) * wave_time**2 / wave_fwhm**2)

    # Wavelet spectrum
    waveX = np.fft.fft(csw * gauss, n_conv)
    waveX /= np.max(waveX)

    # Convolution
    conv_res = np.fft.ifft(signalX * waveX)
    conv_res = conv_res[half_w:-half_w]

    # Extract power
    tf[fi, :] = np.abs(conv_res) ** 2

plt.figure()
plt.contourf(time, freqs, tf, 40, cmap=cmap)
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.show()

# %%
