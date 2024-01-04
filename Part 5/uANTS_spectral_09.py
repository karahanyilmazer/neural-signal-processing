# %%
#   COURSE: Neural signal processing and analysis: Zero to hero
#  SESSION: Static spectral analyses
#  TEACHER: Mike X Cohen, sincxpress.com
#    VIDEO: Spectral analysis of resting-state EEG

# !%matplotlib inline
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

# Set figure settings
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import set_fig_dpi, set_style

set_fig_dpi(), set_style(notebook=True)

# %% The goal of this cell is to plot a power spectrum of resting-state EEG data
mat_file = loadmat(os.path.join('..', 'data', 'EEGrestingState.mat'))
data = mat_file['eegdata'][0]
srate = mat_file['srate'][0][0]


# Create a time vector that starts from 0
n_samples = len(data)
time = np.arange(n_samples) / srate

# Plot the time-domain signal
plt.figure()
plt.plot(time, data)
plt.xlabel('Time (s)')
plt.ylabel('Voltage ($\mu$V)')
plt.show()

# Static spectral analysis
hz = np.linspace(0, srate / 2, int(np.floor(n_samples / 2)) + 1)
amp = 2 * np.abs(np.fft.fft(data) / n_samples)
power = amp ** (1 / 2)

plt.figure()
plt.plot(hz, power[: len(hz)], label='Power')
plt.plot(hz, amp[: len(hz)], label='Amplitude')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude or Power')
plt.legend()
plt.xlim([0, 30])
plt.ylim([0, 2])
plt.show()

# Q: What are the three most prominent features of the EEG spectrum?
# A: 1/f pink noise, alpha peak around 10 Hz and line noise at 50 Hz and its harmonics.
# Q: What do you notice about the difference between the amplitude and power spectra?
# A: The amplitude spectrum highlights the subtle features of the signal, whereas
#    the power spectrum highlights the prominent features of the signal.
# Q: Can you see the ~10 Hz oscillation in the raw time series data?
# A: Yes, but it is too subtle to see without zooming in.

# %%
