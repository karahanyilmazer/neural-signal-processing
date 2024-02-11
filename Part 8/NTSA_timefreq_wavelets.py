# %%
#     COURSE: Solved problems in neural time series analysis
#    SECTION: Time-frequency analyses
#    TEACHER: Mike X Cohen, sincxpress.com
#      VIDEO: Create a family of complex Morlet wavelets

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
# Time parameters
srate = 1025
n_pnts = 2001  # Use an odd number! (To have an exact center point)
time = np.linspace(-1, 1, n_pnts) * (n_pnts - 1) / srate / 2

# wavelet parameters
min_freq = 2  # Hz
max_freq = 54  # Hz
n_freqs = 99

# vector of wavelet frequencies
freqs = np.logspace(np.log10(min_freq), np.log10(max_freq), n_freqs)

# Gaussian parameters
n_cycs = np.logspace(np.log10(3), np.log10(15), n_freqs)
fwhms = np.logspace(np.log10(1), np.log10(0.3), n_freqs)

# Now create the wavelets

wave_fam = np.zeros((2, n_freqs, n_pnts), dtype=complex)

for fi in range(n_freqs):
    # Create complex sine wave
    csw = np.exp(1j * 2 * np.pi * freqs[fi] * time)

    # Create the two Gaussians
    s = n_cycs[fi] / (2 * np.pi * freqs[fi])
    gauss1 = np.exp(-(time**2) / (2 * s) ** 2)

    gauss2 = np.exp(-4 * np.log(2) * time**2 / fwhms[fi] ** 2)

    # Now create the complex Morlet wavelets
    wave_fam[0, fi, :] = csw * gauss1
    wave_fam[1, fi, :] = csw * gauss2

# %% Image the wavelets

# Image the wavelets
typename = ['numcyc', 'FWHM']

plt.figure(figsize=(15, 10))

for typei in range(2):
    for part, title_addon in zip(
        [np.real, np.imag, np.abs], ['Real Part', 'Imag Part', 'Magnitude']
    ):
        plt.subplot(2, 3, typei * 3 + (np.real, np.imag, np.abs).index(part) + 1)
        plt.contourf(time, freqs, part(wave_fam[typei, :, :]), 40, cmap=cmap)
        plt.clim([-1, 1])
        plt.title(f'{typename[typei]}: {title_addon}')
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')
        plt.colorbar()

plt.tight_layout()
plt.show()

# Show an example of one wavelet
plt.figure(figsize=(10, 8))

for typei in range(2):
    plt.subplot(2, 1, typei + 1)
    plt.plot(time, np.real(wave_fam[typei, 40, :]), 'b', label='real')
    plt.plot(time, np.imag(wave_fam[typei, 40, :]), 'r', label='imag')
    plt.plot(time, np.abs(wave_fam[typei, 40, :]), 'k', label='abs')
    plt.legend()
    plt.title(f'{typename[typei]} Wavelet at {freqs[40]:.2f} Hz')
    plt.xlabel('Time (s)')

plt.tight_layout()
plt.show()

# %%
