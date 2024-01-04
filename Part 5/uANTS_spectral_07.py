# %%
#   COURSE: Neural signal processing and analysis: Zero to hero
#  SESSION: Static spectral analyses
#  TEACHER: Mike X Cohen, sincxpress.com
#    VIDEO: Fourier coefficients as complex numbers

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
# Fourier coefficients are difficult to interpret 'numerically', that is, it's difficult
# to extract the information in a Fourier coefficient simply by looking at the real and
# imaginary parts printed out.
# Instead, you can understand them by visualizing them (in the next cell!).


srate = 1000
time = np.arange(srate) / srate
freq = 6

# Create sine waves that differ in power and phase
sine1 = 3 * np.cos(2 * np.pi * freq * time + 0)
sine2 = 2 * np.cos(2 * np.pi * freq * time + np.pi / 6)
sine3 = 1 * np.cos(2 * np.pi * freq * time + np.pi / 3)

# Compute Fourier coefficients
fCoefs1 = np.fft.fft(sine1) / len(time)
fCoefs2 = np.fft.fft(sine2) / len(time)
fCoefs3 = np.fft.fft(sine3) / len(time)

hz = np.linspace(0, srate / 2, int(np.floor(len(time) / 2)) + 1)

# Find the frequency of our sine wave
idx_freq = np.argmin(np.abs(hz - freq))

# Let's look at the coefficients for this frequency
print('6 Hz Fourier coefficient for sin1:', fCoefs1[idx_freq])
print('6 Hz Fourier coefficient for sin2:', fCoefs2[idx_freq])
print('6 Hz Fourier coefficient for sin3:', fCoefs3[idx_freq])

# %% complex numbers as vectors in a polar plot

### Explore the concept that the Fourier coefficients are complex numbers,
#   and can be represented on a complex plane.
#   The magnitude is the length of the line, and the phase is the angle of that line.


# Make polar plots of Fourier coefficients

fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})

ax.plot(
    [0, np.angle(fCoefs1[idx_freq])], [0, 2 * np.abs(fCoefs1[idx_freq])], label='sine1'
)
ax.plot(
    [0, np.angle(fCoefs2[idx_freq])], [0, 2 * np.abs(fCoefs2[idx_freq])], label='sine2'
)
ax.plot(
    [0, np.angle(fCoefs3[idx_freq])], [0, 2 * np.abs(fCoefs3[idx_freq])], label='sine3'
)

ax.legend()

# Set the title or make other adjustments as necessary
ax.set_title("Polar Plot Example")

plt.show()
# %% Extract phase and power information via Euler's formula

# Extract amplitude using Pythagorian theorem
amp11 = np.sqrt(fCoefs1.real**2 + fCoefs1.imag**2)
amp12 = np.sqrt(fCoefs2.real**2 + fCoefs2.imag**2)
amp13 = np.sqrt(fCoefs3.real**2 + fCoefs3.imag**2)

# Extract amplitude using the function abs
amp21 = np.abs(fCoefs1)
amp22 = np.abs(fCoefs2)
amp23 = np.abs(fCoefs3)

# Yet another possibility (the complex number times its conjugate)
amp31 = np.sqrt(fCoefs1 * np.conj(fCoefs1))
amp32 = np.sqrt(fCoefs2 * np.conj(fCoefs2))
amp33 = np.sqrt(fCoefs3 * np.conj(fCoefs3))

# Check if all alternatives are the same
print(np.allclose(amp11, amp21, amp31))
print(np.allclose(amp12, amp22, amp32))
print(np.allclose(amp13, amp23, amp33))

amp1, amp2, amp3 = amp11, amp12, amp13  # Reassign the variables for easier access

# %% And now for phase...

# Extract phase angles using trigonometry
phs11 = np.arctan2(fCoefs1.imag, fCoefs1.real)
phs12 = np.arctan2(fCoefs2.imag, fCoefs2.real)
phs13 = np.arctan2(fCoefs3.imag, fCoefs3.real)

# Extract phase angles using Matlab function angle
phs21 = np.angle(fCoefs1)
phs22 = np.angle(fCoefs2)
phs23 = np.angle(fCoefs3)

print(np.allclose(phs11, phs21))
print(np.allclose(phs12, phs22))
print(np.allclose(phs13, phs23))

phs1, phs2, phs3 = phs11, phs12, phs13

# %%
fig, axs = plt.subplots(2, 1)

# Plot amplitude spectrum
axs[0].plot(hz, 2 * amp1[: len(hz)], label='sine1')
axs[0].plot(hz, 2 * amp2[: len(hz)], label='sine2')
axs[0].plot(hz, 2 * amp3[: len(hz)], label='sine3')
axs[0].set_xlim(freq - 3, freq + 3)
axs[0].set_xlabel('Frequency (Hz)')
axs[0].set_ylabel('Amplitude')
axs[0].legend()

# Plot phase spectrum
axs[1].plot(hz, phs1[: len(hz)], label='sine1')
axs[1].plot(hz, phs2[: len(hz)], label='sine2')
axs[1].plot(hz, phs3[: len(hz)], label='sine3')
axs[1].set_xlim(freq - 3, freq + 3)
axs[1].set_xlabel('Frequency (Hz)')
axs[1].set_ylabel('Phase (rad.)')
axs[1].legend()

plt.tight_layout()
plt.show()


# %%
