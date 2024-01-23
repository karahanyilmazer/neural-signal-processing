# %%
#   COURSE: Neural signal processing and analysis: Zero to hero
#  SECTION: Time-frequency analysis
#  TEACHER: Mike X Cohen, sincxpress.com
#    VIDEO: The five steps of convolution

# !%matplotlib inline
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import convolve1d

# Set figure settings
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import set_fig_dpi, set_style

set_fig_dpi(), set_style()

# %%
# This is the same signal and kernel as used before, but we will implement
# convolution differently.

# Make a kernel (e.g., Gaussian)
kernel = np.exp(-np.linspace(-2, 2, 20) ** 2)
kernel = kernel / sum(kernel)

# Try these options
# kernel = -kernel
# kernel = np.concatenate([np.zeros(9), [1, -1], np.zeros(9)])  # Edge detector!

# Create the signal
signal = np.concatenate(
    [
        np.zeros(30),
        np.ones(2),
        np.zeros(20),
        np.ones(30),
        2 * np.ones(10),
        np.zeros(30),
        -np.ones(10),
        np.zeros(40),
    ]
)

# %% Plot
plt.figure()
plt.plot(kernel + 1, 'b', label='Kernel')
plt.plot(signal, 'k', label='Signal')

# Use the scipy convolve1d function as numpy convolve shifts the results by one sample
# plt.plot(np.convolve(signal, kernel, 'same'), 'r', label='Convolved')
plt.plot(convolve1d(signal, kernel), 'r', label='Convolved')

plt.xlim([0, len(signal)])

# Now for convolution via spectral multiplication

# Step 1: N's of convolution
n_data = len(signal)
n_kern = len(kernel)
n_conv = n_data + n_kern - 1  # Length of result of convolution
half_k = n_kern // 2

# Step 2: FFTs
dataX = np.fft.fft(signal, n=n_conv)  # Important: make sure to properly zero-pad!
kernX = np.fft.fft(kernel, n=n_conv)

# Step 3: Multiply spectra
conv_resX = dataX * kernX

# Step 4: IFFT
conv_res = np.fft.ifft(conv_resX).real

# Step 5: Cut off "wings"
conv_res = conv_res[half_k : -half_k + 1]

# plt.plot(convolve(signal, kernel, 'same'))
# And plot for confirmation!
plt.plot(conv_res, 'go', label='Spectral Multiplication')

plt.legend()
plt.show()

# Q: Is the order of multiplication in step 3 important?
#    Go back to the previous cell and swap the order in the conv() function.
#    Is that order important? Why might that be?
# A: No, the element-wise multiplication is commutative.
#    The order of the conv() parameters however is important if mode='same' is used.

# %%
