# %%
#   COURSE: Neural signal processing and analysis: Zero to hero
#  SECTION: Time-frequency analysis
#  TEACHER: Mike X Cohen, sincxpress.com
#    VIDEO: Complex Morlet wavelet convolution

# !%matplotlib qt
import os
import sys

import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import numpy as np
from matplotlib.projections import PolarAxes
from scipy.io import loadmat

# Set figure settings
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import set_fig_dpi, set_style

set_fig_dpi(), set_style()
# %%
# Now you will observe convolution with real data
mat = loadmat(os.path.join('..', 'data', 'v1_laminar.mat'))
# Get the time points
time_vec = mat['timevec'][0]
# Get the sampling frequency
srate = mat['srate'][0][0]

# Extract a single trial from a channel
data = mat['csd'][5, :, 9]

# Create a complex Morlet wavelet
# Note the alternative method for creating centered time vector
time = np.arange(0, 2 * srate) / srate
time = time - np.mean(time)
freq = 45  # Frequency of wavelet, in Hz

# Create Gaussian window
s = 7 / (2 * np.pi * freq)  # Using num-cycles formula
cmw = np.exp(1j * 2 * np.pi * freq * time) * np.exp(-(time**2) / (2 * s**2))


# %% Now for convolution

# Step : N's of convolution
n_data = len(time_vec)
n_kern = len(cmw)
n_conv = n_data + n_kern - 1
half_k = int(np.floor(n_kern / 2))

# Step 2: FFTs
dataX = np.fft.fft(data, n_conv)
kernX = np.fft.fft(cmw, n_conv)

# Step 2.5: Normalize the wavelet (try it without this step!)
kernX = kernX / np.max(kernX)

# Step 3: Multiply spectra
conv_resX = dataX * kernX

# Step 4: IFFT
conv_res = np.fft.ifft(conv_resX)

# Step 5: Cut off "wings"
conv_res = conv_res[half_k : -half_k + 1]

# %% Now for plotting

# Compute hz for plotting
hz = np.linspace(0, srate / 2, int(np.floor(n_conv / 2)) + 1)

plt.figure()
plt.subplot(211)

# Plot power spectrum of data
plt.plot(hz, np.abs(dataX[: len(hz)]), label='Data Spectrum')

# Plot power spectrum of wavelet
plt.plot(
    hz, np.abs(kernX[: len(hz)]) * np.max(np.abs(dataX)) / 2, label='Wavelet Spectrum'
)

# Plot power spectrum of convolution result
plt.plot(hz, np.abs(conv_resX[: len(hz)]), label='Convolution Result')
plt.xlim([0, freq * 2])
plt.xlabel('Frequency (Hz)'), plt.ylabel('Amplitude (a.u.)')
plt.legend()


# Now plot in the time domain
plt.subplot(212)
plt.plot(time_vec, data, label='LFP Data')
plt.plot(time_vec, conv_res.real, label='Convolution Result')
plt.xlim([-0.1, 1.3])
plt.legend()
plt.xlabel('Time (s)')
plt.ylabel('Activity ($\mu$V)')

plt.tight_layout()
plt.show()

# %% Extracting the three features of the complex wavelet result

plt.figure()

# Plot the filtered signal (projection onto real axis)
plt.subplot(311)
plt.plot(time_vec, conv_res.real)
plt.ylabel('Amplitude ($\mu$V)')
plt.xlim(-0.1, 1.4)


# Plot power (squared magnitude from origin to dot-product location in complex space)
plt.subplot(312)
plt.plot(time_vec, np.abs(conv_res) ** 2)
plt.ylabel('Power ($\mu V^2$)')
plt.xlim(-0.1, 1.4)


# Plot phase (angle of vector to dot-product, relative to positive real axis)
plt.subplot(313)
plt.plot(time_vec, np.angle(conv_res))
plt.xlabel('Time (ms)')
plt.ylabel('Phase (rad.)')
plt.xlim(-0.1, 1.4)

plt.tight_layout()
plt.show()

# %% Viewing the results as a movie
plt.figure()

# Setup the time course plot
ax_time = plt.subplot(212)
(line,) = plt.plot(time_vec, np.abs(conv_res), 'k', linewidth=2)
plt.xlim(time_vec[0], time_vec[-1])
plt.ylim(np.min(np.abs(conv_res)), np.max(np.abs(conv_res)))
plt.xlabel('Time (sec.)')
plt.ylabel('Amplitude ($\mu$V)')


# Function to create a polar plot
def polar_plot(ax, theta, r):
    ax = plt.subplot(ax, polar=True)
    ax.plot(theta, r, 'k')
    return ax


ax_polar = plt.subplot(211, polar=True)

# Loop to update the plots
for ti in range(0, len(time_vec), 5):
    # Draw complex values in polar space
    ax_polar.plot(
        np.angle(conv_res[max(0, ti - 100) : ti]),
        np.abs(conv_res[max(0, ti - 100) : ti]),
    )
    ax_polar.text(-0.75, 0, f'{np.round(1000*time_vec[ti])} ms')

    # Now show in 'linear' plot
    line.set_xdata(time_vec[max(0, ti - 100) : ti])
    line.set_ydata(np.abs(conv_res[max(0, ti - 100) : ti]))

    plt.draw()
    plt.pause(0.1)

    # Clear the polar plot for the next iteration
    # ax_polar.cla()
