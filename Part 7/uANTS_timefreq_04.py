# %%
#   COURSE: Neural signal processing and analysis: Zero to hero
#  SECTION: Time-frequency analysis
#  TEACHER: Mike X Cohen, sincxpress.com
#    VIDEO: Convolve real data with a Gaussian

# !%matplotlib inline
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

# Set figure settings
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import set_fig_dpi, set_style

set_fig_dpi(), set_style()
# %%
# Now you will observe convolution with real data
mat = loadmat(os.path.join('..', 'data', 'v1_laminar.mat'))
# Get the time points
time = mat['timevec'][0]
# Get the epoched data of shape (channels x samples x trials)
data = mat['csd']
# Get the sampling frequency
srate = mat['srate'][0][0]

# Signal will be ERP from channel 7
signal = np.mean(data[6, :, :], 1)

# Create a Gaussian
h = 0.012  # FWHM in seconds

g_time = np.arange(-1, 1, 1 / srate)
gauss = np.exp(-4 * np.log(2) * g_time**2 / h**2)
gauss = gauss / sum(gauss)  # Amplitude normalization

# Run convolution
# Step 1: N's of convolution
n_data = len(signal)
n_kern = len(gauss)
n_conv = n_data + n_kern - 1  # Length of result of convolution
half_k = n_kern // 2

# Step 2: FFTs
dataX = np.fft.fft(signal, n=n_conv)  # Important: Make sure to properly zero-pad!
kernX = np.fft.fft(gauss, n=n_conv)

# Step 3: Multiply spectra
conv_resX = dataX * kernX

# Step 4: IFFT
conv_res = np.fft.ifft(conv_resX).real

# Step 5: Cut off "wings"
conv_res = conv_res[half_k : -half_k + 1]

plt.figure()
plt.plot(time, signal, label='Original ERP')
plt.plot(time, conv_res, label='Gaussian-Convolved')
plt.xlim(-0.1, 1.4)
plt.xlabel('Time (s)')
plt.ylabel('Activity ($\mu V$)')
plt.show()

# Q: What is the effect of changing the h parameter?
#    What value leaves the ERP mostly unchanged?
#    What value makes the ERP unrecognizable?
#    What range of values seems "good"?
#    (philosophical) What does the answer to the previous question
#    tell you about the temporal precision of V1 ERP?
# A: Larger FWHM value --> wider time Gaussian --> narrower frequency Gaussian
#    With h = 0.05 ERP stays roughly the same
#    With h = 0.1 ERP is unrecognizable
#    Probably h = 0.007 to h = 0.012 is "good" enough
#    The temporal resolution is technically the same but with larger h values, the
#    interesting gamma activity gets supressed.

# %% Show the mechanism of convolution (spectral multiplication)

hz = np.linspace(0, srate, n_conv)

plt.figure()
# Normalized for visualization
plt.plot(hz, abs(dataX) / max(abs(dataX)), label='Original Signal')
plt.plot(hz, abs(kernX), label='Kernel')
plt.plot(hz, abs(dataX * kernX) / max(abs(dataX)), label='Convolution Result')

plt.xlim(0, 150)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude (norm.)')
plt.legend()
plt.show()

# %%
