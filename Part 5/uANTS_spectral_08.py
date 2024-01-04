# %%
#   COURSE: Neural signal processing and analysis: Zero to hero
#  SESSION: Static spectral analyses
#  TEACHER: Mike X Cohen, sincxpress.com
#    VIDEO: Positive/negative spectrum amplitude scaling

# !%matplotlib qt
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

# Set figure settings
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import set_fig_dpi, set_style

set_fig_dpi(), set_style(notebook=True)

# %%
# This cell will run a video that shows the complex sine waves used
# in the Fourier transform. The blue line corresponds to the real part
# and the green line corresponds to the imaginary part. The title tells
# you the frequency as a fraction of the sampling rate (Nyquist = 0.5).
# Notice what happens as the sine waves go into the negative frequencies.

# Setup parameters
N = 100
fTime = np.arange(N) / N

# Generate the plot
fig, ax = plt.subplots()
(line_r,) = ax.plot(fTime, np.ones(N))
(line_i,) = ax.plot(fTime, np.ones(N), '--')
ax.set_xlabel('Time (norm.)')
ax.set_ylabel('Amplitude')
title = ax.set_title('')

# Loop over frequencies
for fi in range(1, N + 1):
    # Create sine wave for this frequency
    fourierSine = np.exp(-1j * 2 * np.pi * (fi - 1) * fTime)

    # Update graph
    line_r.set_ydata(np.real(fourierSine))
    line_i.set_ydata(np.imag(fourierSine))
    which_half = 'negative' if fi < N / 2 else 'positive'
    title.set_text(f'{fi/N} of $f_s$ ({which_half})')
    plt.ylim([-1.5, 1.5])
    plt.pause(0.1)

plt.show()

# Q: What happens to the sine waves as they approach f=0.5?
# A: They get faster and faster.
# Q: What happens to the sine waves above f=0.5?
# A: They start getting slower.

# %% Scaling of Fourier coefficients
#   The goal of this section of code is to understand the necessity and logic
#   behind the two normalization factors that get the Fourier coefficients
#   in the same scale as the original data.

# Create the signal
srate = 1000  # hz
time = np.arange(3 * srate) / srate  # time vector in seconds
pnts = len(time)  # number of time points
signal = 2.5 * np.sin(2 * np.pi * 4 * time)

# Prepare the Fourier transform
four_time = np.arange(pnts) / pnts
four_coefs = np.zeros_like(signal, dtype=complex)

for fi in range(pnts):
    # Create complex sine wave and compute dot product with signal
    csw = np.exp(-1j * 2 * np.pi * fi * four_time)
    four_coefs[fi] = np.dot(signal, csw)

# Extract amplitudes
amps = np.abs(four_coefs)
# amps = 2 * np.abs(four_coefs / pnts)  # Properly scaled amplitudes

# Compute frequencies vector
# hz = np.linspace(0, int(srate / 2), int(np.floor(pnts / 2)) + 1)  # Alternative
hz = np.linspace(0, srate / 2, pnts // 2 + 1)

plt.figure()
plt.stem(hz, amps[: len(hz)])
plt.xlim([0, 10])
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude (a.u.)')
plt.show()

# Q: Does the amplitude of 4 Hz in the figure match the simulated signal?
# A: No, the amplitude should be 2.5 but it is more than 3500.
# Q: Does the amplitude also depend on the length of the signal?
#    Apply the two normalization factors discussed in lecture.
#    Test whether the scaling is robust to signal length.
# A: Yes, it does. The scaling factors prevent this.

# %% DC reflects the mean offset

# NOTE: below is the same signal with (1) small, (2) no, (3) large DC
#       Is the amplitude spectrum accurate at 0 Hz???
signalX1 = np.fft.fft(signal + 2) / pnts
signalX2 = np.fft.fft(signal - np.mean(signal)) / pnts
signalX3 = np.fft.fft(signal + 10) / pnts


fig, axs = plt.subplots(2, 1)
# Plot signals in the time domain
axs[0].plot(np.fft.ifft(signalX1).real * pnts)
axs[0].plot(np.fft.ifft(signalX2).real * len(signal))
axs[0].plot(np.fft.ifft(signalX3).real * pnts)
axs[0].set_xlabel('Time (ms)')
axs[0].set_ylabel('Amplitude')

# Multiply all the amplitudes by 2 (except DC)
ampl1 = np.abs(signalX1[: len(hz)])
ampl1[1:] = ampl1[1:] * 2
ampl2 = np.abs(signalX2[: len(hz)])
ampl2[1:] = ampl2[1:] * 2
ampl3 = np.abs(signalX3[: len(hz)])
ampl3[1:] = ampl3[1:] * 2

# Plot signals in the frequency domain
axs[1].plot(hz, ampl1, 'o-', label='+2 mean')
axs[1].plot(hz, ampl2, 'd-', label='de-meaned')
axs[1].plot(hz, ampl3, '*-', label='+10 mean')

# Plot signals in the frequency domain
axs[1].set_xlabel('Frequencies (Hz)')
axs[1].set_ylabel('Amplitude')
axs[1].legend()
axs[1].set_xlim([0, 10])

plt.tight_layout()
plt.show()

# Q: Can you adapt the code for accurate scaling of all three signals?
#    (Yes, of course you can! Do it!)
# A: Done.

# %%
