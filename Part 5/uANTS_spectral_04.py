# %%
#   COURSE: Neural signal processing and analysis: Zero to hero
#  SESSION: Static spectral analyses
#  TEACHER: Mike X Cohen, sincxpress.com
#    VIDEO: Complex sine waves
#     GOAL: Create a complex-valued sine wave

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
# General simulation parameters
srate = 500  # Sampling rate in Hz
time = np.arange(0, 2, 1 / srate)  # Time in seconds

# Sine wave parameters
freq = 5  # Frequency in Hz
amp = 2  # Amplitude in a.u.
phase = np.pi / 3  # Phase in radians

# Generate the complex sine wave
csw = amp * np.exp(1j * (2 * np.pi * freq * time + phase))


# Plot in 2D
fig = plt.figure()
ax1 = fig.add_subplot(211)
ax1.plot(time, csw.real, label='real')
ax1.plot(time, csw.imag, label='imag')
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Amplitude  (u.a.)')
ax1.set_title('Complex Sine Wave Projections')
ax1.legend()

# Plot in 3D
ax2 = fig.add_subplot(212, projection='3d')
ax2.plot3D(time, csw.real, csw.imag)
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Real Part')
ax2.set_zlabel('Imag Part')
ax2.set_ylim(np.array([-1, 1]) * amp * 3)
ax2.set_zlim(np.array([-1, 1]) * amp * 3)

plt.tight_layout()
plt.show()

# %%
