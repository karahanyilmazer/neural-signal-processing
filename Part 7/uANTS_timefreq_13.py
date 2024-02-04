# %%
#   COURSE: Neural signal processing and analysis: Zero to hero
#  SECTION: Time-frequency analysis
#  TEACHER: Mike X Cohen, sincxpress.com
#    VIDEO: Filter-Hilbert

# !%matplotlib inline
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from scipy.fft import fft
from scipy.signal import filtfilt, firls, hilbert

# Set figure settings
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import set_fig_dpi, set_style

set_fig_dpi(), set_style()
# %%
# Narrowband filtering via FIR

# Filter parameters
srate = 1024  # Hz
nyquist = srate / 2
f_range = [20, 25]
trans_w = 0.1
order = np.round(3 * srate / f_range[0]).astype(int)
order = order + 1 if order % 2 == 0 else order  # Convert to an even number (for firls)

gains = [0, 0, 1, 1, 0, 0]
freqs = (
    np.array(
        [
            0,
            f_range[0] - f_range[0] * trans_w,
            *f_range,
            f_range[1] + f_range[1] * trans_w,
            nyquist,
        ]
    )
    / nyquist
)

# Filter kernel
filt_kern = firls(order, freqs, gains)

# Compute the power spectrum of the filter kernel
hz = np.linspace(0, srate / 2, len(filt_kern) // 2 + 1)
filt_pow = np.abs(fft(filt_kern))  # ** 2
filt_pow = filt_pow[: len(hz)]

fig = plt.figure()
gs = GridSpec(2, 2, figure=fig)
ax1 = fig.add_subplot(gs[0, :])
ax2 = fig.add_subplot(gs[1, 0])
ax3 = fig.add_subplot(gs[1, 1])
ax4 = fig.add_subplot(gs[1, :])

# Plot the filter kernel
ax1.plot(filt_kern)
ax1.set_title('Filter Kernel')
ax1.set_xlabel('Time Points')

# Plot amplitude spectrum of the filter kernel
ax2.plot(hz, filt_pow, 'ks-', markerfacecolor='w', label='Actual')
ax2.plot(freqs * nyquist, gains, 'ro-', markerfacecolor='w', label='Ideal')
ax2.set_xlim(0, f_range[0] * 4)
ax2.set_xlabel('Frequency (Hz)')
ax2.set_ylabel('Filter Gain')
ax2.legend()

# Plot filter gain in dB
ax3.axvline(f_range[0], color='grey', linestyle=':')
ax3.plot(hz, 10 * np.log10(filt_pow), 'ks-', markersize=10, markerfacecolor='w')
ax3.set_xlim(0, f_range[0] * 4)
ax3.set_ylim(-50, 2)
ax3.set_xlabel('Frequency (Hz)')
ax3.set_ylabel('Filter Gain (dB)')

ax4.set_title('Frequency Response')
ax4.axis('off')

plt.tight_layout()
plt.show()

# Q: Is this a good filter? The answer is yes if there is a good
#    match between the "ideal" and "actual" spectral response.
# A: Nope, the actual filter is not very close to the ideal filter.
# Q: One important parameter is the order (number of points in the
#    kernel). Based on your knowledge of the Fourier transform,
#    should this parameter be increased or decreased to get a
#    better filter kernel? First answer, then try it!
# A: Increased. Increasing the number of cycles from 3 to 10 does the trick.

# %%
# Apply the filter to random noise

# Generate random noise as "signal"
signal = np.random.randn(srate * 4)

# Apply the filter kernel to the signal
filt_sig = filtfilt(filt_kern, 1, signal)

fig = plt.figure()
gs = GridSpec(1, 4, figure=fig)
ax1 = fig.add_subplot(gs[0, 0:3])
ax2 = fig.add_subplot(gs[0, 3])

# Plot time series
ax1.plot(signal, 'r', label='Original')
ax1.plot(filt_sig, 'k', label='Filtered')
ax1.set_xlim(1, len(signal))
ax1.set_title('Time Domain')
ax1.set_xlabel('Time (a.u.)')
ax1.set_ylabel('Amplitude (a.u.)')
ax1.legend()

# Plot power spectrum
hz = np.linspace(0, srate, len(signal))
ax2.plot(hz, np.abs(fft(signal)), 'r')
ax2.plot(hz, np.abs(fft(filt_sig)), 'k')
ax2.set_xlim(0, f_range[1] * 2)
ax2.set_title('Frequency Domain')
ax2.set_xlabel('Frequency (Hz)')

plt.tight_layout()
plt.show()

# %%
# The Hilbert transform

# Take the Hilbert transform
hilb_filt_sig = hilbert(filt_sig)

plt.figure()
plt.subplot(311)
plt.plot(np.real(hilb_filt_sig))
plt.title('Real part of Hilbert')

plt.subplot(312)
plt.plot(np.abs(hilb_filt_sig))
plt.title('Magnitude of Hilbert')

plt.subplot(313)
plt.plot(np.angle(hilb_filt_sig))
plt.title('Angle of Hilbert')

plt.tight_layout()
plt.show()

# %%
