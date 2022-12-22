# %%
#     COURSE: Solved challenges in neural time series analysis
#    SECTION: Simulating EEG data
#      VIDEO: Non-stationary narrowband activity via filtered noise
# Instructor: sincxpress.com

import numpy as np
import matplotlib.pyplot as plt

# %% SIMULATION DETAILS
pnts = 4567
srate = 987

# Signal parameters in Hz
peakfreq = 14
fwhm = 5

# Frequencies
hz = np.linspace(0, srate, pnts)

# %%
# Create frequency-domain Gaussian
s = fwhm * (2 * np.pi - 1) / (4 * np.pi)  # Normalized width
x = hz - peakfreq                         # Shifted frequencies
fg = np.exp(-0.5 * (x / s)**2)            # Gaussian

# Fourier coefficients of random spectrum
fc = np.random.rand(pnts) * np.exp(1j * 2 * np.pi * np.random.rand(pnts))

# Taper with Gaussian
fc = fc * fg

# Go back to time domain to get EEG data
signal = 2 * np.fft.ifft(fc).real

# %% PLOTTING

mosaic = [['freq', 'freq'], ['time', 'time']]
fig = plt.figure(constrained_layout=True)
ax_dict = fig.subplot_mosaic(mosaic)

ax_dict['freq'].plot(hz, abs(fc))
ax_dict['freq'].set_xlim(0, peakfreq * 3)
ax_dict['freq'].set_xlabel('Frequency (Hz)')
ax_dict['freq'].set_ylabel('Amplitude (a.u.)')
ax_dict['freq'].set_title('Frequency domain')

ax_dict['time'].plot(np.arange(pnts) / srate, signal)
ax_dict['time'].set_title('Time domain')
ax_dict['time'].set_xlabel('Time (s)')
ax_dict['time'].set_ylabel('Amplitude')

plt.show()

# %%
