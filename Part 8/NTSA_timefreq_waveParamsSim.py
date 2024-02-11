# %%
#     COURSE: Solved problems in neural time series analysis
#    SECTION: Time-frequency analyses
#    TEACHER: Mike X Cohen, sincxpress.com
#      VIDEO: Exploring wavelet parameters in simulated data

# !%matplotlib inline
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fft, ifft
from scipy.signal import hilbert

# Set figure settings
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import get_cmap, set_fig_dpi, set_style

set_fig_dpi(), set_style()
cmap = get_cmap('parula')

# %% Simulate data as transient oscillation

pnts = 4000
srate = 1000
stime = np.arange(0, pnts) / srate - 2

# Gaussian parameters
fwhm = 0.4

# Sine wave parameters
sine_freq = 10

# Create signal
gaus = np.exp(-(4 * np.log(2) * stime**2) / fwhm**2)
cosw = np.cos(2 * np.pi * sine_freq * stime + 2 * np.pi * np.random.rand())
signal = cosw * gaus

# Get signal amplitude
sig_amp = np.abs(hilbert(signal))

plt.figure()
plt.plot(stime, signal, 'k')
plt.plot(stime, sig_amp, 'r--', linewidth=1)
plt.xlabel('Time (s)')
plt.xlabel('Amplitude (a.u.)')

plt.show()

# %% Comparing fixed number of wavelet cycles

# Wavelet parameters
num_freqs = 50
min_freq = 2
max_freq = 20

# Set a few different wavelet widths (FWHM parameter)
fwhms = [0.1, 0.5, 2]

# Other wavelet parameters
freqs = np.linspace(min_freq, max_freq, num_freqs)
w_time = np.arange(-2, 2 + 1 / srate, 1 / srate)
half_wave = (len(w_time) - 1) // 2

# FFT parameters
n_kern = len(w_time)
n_conv = n_kern + pnts - 1

# Initialize output time-frequency data
tf = np.zeros((len(fwhms), len(freqs), pnts))

# FFT of data (doesn't change on frequency iteration)
dataX = fft(signal, n_conv)

# Loop over cycles
for fwhmi in range(len(fwhms)):
    for fi in range(len(freqs)):
        # Create wavelet and get its FFT
        cmw = np.exp(2 * 1j * np.pi * freqs[fi] * w_time) * np.exp(
            -4 * np.log(2) * w_time**2 / fwhms[fwhmi] ** 2
        )
        cmwX = fft(cmw, n_conv)
        cmwX = cmwX / cmwX[np.argmax(np.abs(cmwX))]

        # Run convolution, trim edges
        comp_sig = ifft(cmwX * dataX)
        comp_sig = comp_sig[half_wave:-half_wave]

        # Put power data into big matrix
        tf[fwhmi, fi, :] = np.abs(comp_sig)

# %% Plot results

plt.figure()

# Define the grid
gridsize = (4, 6)  # Rows, columns
ax1 = plt.subplot2grid(gridsize, (0, 0), colspan=2, rowspan=2)
ax2 = plt.subplot2grid(gridsize, (0, 2), colspan=2, rowspan=2)
ax3 = plt.subplot2grid(gridsize, (0, 4), colspan=2, rowspan=2)
ax4 = plt.subplot2grid(gridsize, (2, 0), colspan=3, rowspan=2)
ax5 = plt.subplot2grid(gridsize, (2, 3), colspan=3, rowspan=2)

# Time-frequency plots
ax1.contourf(stime, freqs, tf[0, :, :], 40, cmap=cmap)
ax1.set_title(f'{fwhms[0]} s FWHM')
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Frequency (Hz)')

ax2.contourf(stime, freqs, tf[1, :, :], 40, cmap=cmap)
ax2.set_title(f'{fwhms[1]} s FWHM')
ax2.set_xlabel('Time (s)')

ax3.contourf(stime, freqs, tf[2, :, :], 40, cmap=cmap)
ax3.set_title(f'{fwhms[2]} s FWHM')
ax3.set_xlabel('Time (s)')


# Show amplitude envelopes at peak frequency
f_idx = np.argmin(np.abs(freqs - sine_freq))
ax4.plot(stime, tf[:, f_idx, :].T)
ax4.plot(stime, sig_amp / 2, 'k--')
ax4.set_title('Time domain')
ax4.set_xlabel('Time (s)')
ax4.set_ylabel('Amplitude (a.u.)')
ax4.legend([str(fwhm) for fwhm in fwhms] + ['Truth'])

# Show spectra at one time point
t_idx = np.argmin(np.abs(stime))
ax5.plot(freqs, tf[:, :, t_idx].T)
ax5.plot(np.linspace(0, srate, pnts), 2 * np.abs(fft(signal) / pnts), 'k--')
ax5.set_xlim([freqs[0], freqs[-1]])
ax5.set_xlabel('Frequency (Hz)')
ax5.set_title('Frequency domain')
ax5.legend([str(fwhm) for fwhm in fwhms] + ['Truth'])

plt.tight_layout()
plt.show()

# %%
