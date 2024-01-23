# %%
#   COURSE: Neural signal processing and analysis: Zero to hero
#  SECTION: Time-frequency analysis
#  TEACHER: Mike X Cohen, sincxpress.com
#    VIDEO: Convolution with all trials!

# !%matplotlib inline
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

# Set figure settings
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import get_cmap, set_fig_dpi, set_style

set_fig_dpi(), set_style()
cmap = get_cmap('parula')
# %%
# This is the "super-trial" concatenation trick you saw in the slides.
# Notice the size of the reshaped data and the new convolution parameters.

# Now you will observe convolution with real data
mat = loadmat(os.path.join('..', 'data', 'v1_laminar.mat'))
# Get the time points
time_vec = mat['timevec'][0]
# Get the sampling frequency
srate = mat['srate'][0][0]

# Extract all the trials from a single channel
data = mat['csd'][5, :, :]
data_shape = data.shape

# Reshape the data to be 1D --> form a "super-trial"
dataR = data.T.reshape(-1)

# Note the alternative method for creating centered time vector
time = np.arange(0, 2 * srate) / srate
time = time - np.mean(time)
freq = 45  # Frequency of wavelet, in Hz

# Create Gaussian window
s = 7 / (2 * np.pi * freq)  # Using num-cycles formula
cmw = np.exp(1j * 2 * np.pi * freq * time) * np.exp(-(time**2) / (2 * s**2))

# %% Now for convolution
# Step 1: N's of convolution
n_data = len(dataR)
n_kern = len(time)
n_conv = n_data + n_kern - 1
half_k = int(np.floor(n_kern / 2))

# Step 2: FFTs
dataX = np.fft.fft(dataR, n_conv)
kernX = np.fft.fft(cmw, n_conv)

# Step 2.5: Normalize the wavelet (try it without this step!)
kernX = kernX / np.max(kernX)

# Step 3: Multiply spectra
conv_resX = dataX * kernX

# Step 4: IFFT
conv_res = np.fft.ifft(conv_resX)

# Step 5: Cut off "wings"
conv_res = conv_res[half_k : -half_k + 1]

# New step 6: Reshape!
conv_res_2D = conv_res.reshape(data.shape, order='F')

# %% Plotting
plt.figure()
plt.subplot(121)
plt.imshow(
    data.T,
    extent=[time_vec[0], time_vec[-1], data.shape[1], 0],
    clim=[-2000, 2000],
    cmap=cmap,
    aspect='auto',
)
plt.xlabel('Time (s)')
plt.ylabel('Trials')
plt.xlim(-0.1, 1.4)
plt.title('Broadband Signal')

plt.subplot(122)
plt.imshow(
    np.abs(conv_res_2D).T,
    extent=[time_vec[0], time_vec[-1], data.shape[1], 0],
    clim=[-500, 500],
    cmap=cmap,
    aspect='auto',
)
plt.xlabel('Time (s)')
plt.xlim(-0.1, 1.4)
plt.title(f'Power Time Series at {freq} Hz')

plt.tight_layout()
plt.show()
# %%
