# %%
#     COURSE: Solved challenges in neural time series analysis
#    SECTION: Simulating EEG data
#      VIDEO: Project 2: dipole-level EEG data
# Instructor: sincxpress.com

# !%matplotlib qt
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mne import create_info
from mne.channels import make_standard_montage
from mne.viz import plot_topomap
from scipy.io import loadmat

# Set figure settings
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import get_cmap, set_fig_dpi, set_style

set_fig_dpi(), set_style()
cmap = get_cmap('parula')


# %%
def plot_EEG(times, data, ch_idx=0):
    # Convert s to ms
    times = times.copy() * 1000

    # Initialize the figure
    mosaic = [['ERP', 'ERP'], ['PSD', 'TF']]
    fig = plt.figure(constrained_layout=True)
    ax_dict = fig.subplot_mosaic(mosaic)

    # Frequencies in Hz for the PSD plot
    freqs_psd = np.linspace(0, srate, n_samples)

    # Frequencies in Hz (hard-coded to 2 to 30 in 40 steps)
    freqs = np.linspace(2, 30, 40)
    # Number of wavelet cycles (hard-coded to 3 to 10)
    waves = 2 * (np.linspace(3, 10, len(freqs)) / (2 * np.pi * freqs)) ** 2

    # Setup wavelet and convolution parameters
    wave_t = np.arange(-2, 2 + 1 / srate, 1 / srate)
    half_w = int(np.floor(len(wave_t) / 2) + 1)
    n_conv = n_samples * n_trials + len(wave_t) - 1

    if data.ndim == 3:
        # ERP
        ax_dict['ERP'].plot(times, data[:, ch_idx, :].T, color='grey', alpha=0.1)
        ax_dict['ERP'].plot(times, np.mean(data[:, ch_idx, :].T, axis=1), color='black')

        # PSD
        # Perform FFT along the columns (trials)
        pw = np.mean(
            (2 * np.abs(np.fft.fft(data[:, ch_idx, :].T, axis=0) / n_samples)) ** 2,
            axis=1,
        )

        # TF
        data_fft = np.fft.fft(data[:, ch_idx, :].T.reshape(1, -1, order='F'), n_conv)
    else:
        # ERP
        ax_dict['ERP'].plot(times, data[ch_idx, :], color='black')

        # PSD
        pw = (2 * np.abs(np.fft.fft(data[ch_idx, :], axis=0) / n_samples)) ** 2

        # TF
        data_fft = np.fft.fft(data[ch_idx, :].reshape(1, -1, order='F'), n_conv)

    # Initialize the time-frequency matrix
    tf_mat = np.zeros((len(freqs), n_samples))

    for i in range(len(freqs)):
        wave_x = np.fft.fft(
            np.exp(2 * 1j * np.pi * freqs[i] * wave_t)
            * np.exp(-(wave_t**2) / waves[i]),
            n_conv,
        )
        wave_x = wave_x / np.max(wave_x)

        amp_spec = np.fft.ifft(wave_x * data_fft)
        amp_spec = amp_spec[0][half_w - 1 : -half_w + 1].reshape(
            (n_samples, n_trials), order='F'
        )

        tf_mat[i, :] = np.mean(np.abs(amp_spec), axis=1)

    ax_dict['ERP'].set_title(f'ERP From Channel {ch_idx+1}')
    ax_dict['ERP'].set_xlabel('Time (ms)')
    ax_dict['ERP'].set_ylabel('Activity')
    ax_dict['ERP'].set_xlim(times.min(), times.max())

    ax_dict['PSD'].plot(freqs_psd, pw)
    ax_dict['PSD'].set_title(f'Static Power Spectrum')
    ax_dict['PSD'].set_xlabel('Frequency (Hz)')
    ax_dict['PSD'].set_ylabel('Power')
    ax_dict['PSD'].set_xlim(0, 40)

    ax_dict['TF'].contourf(times, freqs, tf_mat, 40, cmap=cmap)
    ax_dict['TF'].set_title(f'Time-Frequency Plot')
    ax_dict['TF'].set_xlabel('Time (ms)')
    ax_dict['TF'].set_ylabel('Frequency (Hz)')
    ax_dict['TF'].set_xlim(times.min(), times.max())

    plt.show()


# Load the mat file containing EEG, leadfield and channel locations
mat = loadmat('emptyEEG')
ch_info = []
for ch in mat['EEG'][0][0][21][0]:
    label, theta, radius, x, y, z, sph_theta, sph_phi, sph_radius, _, _, _ = ch
    ch_info.append(
        (
            label[0],
            theta[0][0],
            radius[0][0],
            x[0][0],
            y[0][0],
            z[0][0],
            sph_theta[0][0],
            sph_phi[0][0],
            sph_radius[0][0],
        )
    )

ch_df = pd.DataFrame(
    ch_info,
    columns=[
        'label',
        'theta',
        'radius',
        'x',
        'y',
        'z',
        'sph_theta',
        'sph_phi',
        'sph_radius',
    ],
)

n_chs = len(ch_info)
n_trials = 40
n_samples = 2000
srate = 1024
times = np.arange(n_samples) / srate

gain = mat['lf'][0][0][2]
grid_loc = mat['lf'][0][0][5]

info = create_info(ch_df['label'].to_list(), srate, 'eeg')
montage = make_standard_montage('standard_1020')
info.set_montage(montage)

# %%
# Select dipole location (more-or-less random)
dip_loc = 108

# Plot brain dipoles
fig = plt.figure()
ax1 = plt.subplot(121, projection='3d')
ax2 = plt.subplot(122)
# Scatter plot the channel positions
ax1.scatter(grid_loc[:, 0], grid_loc[:, 1], grid_loc[:, 2], marker='o', s=8, alpha=0.3)
ax1.scatter(
    grid_loc[dip_loc, 0],
    grid_loc[dip_loc, 1],
    grid_loc[dip_loc, 2],
    marker='o',
    s=50,
    color='red',
)

ax1.set_title('Brain Dipole Locations')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('z')

# Each dipole can be projected onto the scalp using the forward model.
# The code below shows this projection from one dipole.
im, _ = plot_topomap(
    -gain[:, 0, dip_loc],
    info,
    names=np.arange(n_chs) + 1,
    axes=ax2,
    cmap=cmap,
    sphere='eeglab',
)
fig.colorbar(im, ax=ax2)
ax2.set_title('Signal Dipole Projection')

plt.show()

# %% Add signal to one dipole and project it to scalp

# Initialize all dipole data
dipole_data = np.zeros((gain.shape[2], n_samples))

# Add signal to one dipole
dipole_data[dip_loc, :] = np.sin(2 * np.pi * 10 * times)

# Now project dipole data to scalp electrodes
data = gain[:, 0, :] @ dipole_data

# Plot the data
plot_EEG(times, data, ch_idx=30)

# %% 1) Pure sine wave with amplitude explorations

# Dipole amplitude magnitude
amp = 1

# Initialize all dipole data
data = np.zeros((n_trials, n_chs, n_samples))
dipole_data = np.zeros((gain.shape[2], n_samples))
dipole_data[dip_loc, :] = amp * np.sin(2 * np.pi * 10 * times)

# Compute one trial
signal = gain[:, 0, :] @ dipole_data

# Repeat that for N trials
for i in range(n_trials):
    data[i, :, :] = signal

# Plot the data
plot_EEG(times, data, ch_idx=30)

# Q: What is the smallest amplitude of dipole signal that still
#    elicits a scalp-level response?
# A: All the other dipoles are 0, so there will always be a response.

# %% 2) Sine wave with noise

# Standard deviation of the noise
noise_std = 1

# Dipole amplitude magnitude
amp = 1

# Initialize all dipole data
data = np.zeros((n_trials, n_chs, n_samples))

for i in range(n_trials):
    dipole_data = np.random.randn(gain.shape[2], n_samples) * noise_std
    dipole_data[dip_loc, :] = amp * np.sin(2 * np.pi * 10 * times)

    signal = gain[:, 0, :] @ dipole_data
    data[i, :, :] = signal

# Plot the data
plot_EEG(times, data, ch_idx=30)

# Q: Given amplitude=1 of dipole signal, what standard deviation of noise
#    at all other dipoles overpowers the signal (qualitatively)?
# A: Already around 1 the peak is industrialized from the noise.

# %% 3) Non-oscillatory transient in one dipole, noise in all other dipoles
# Define the parameters
peak_time = 1  # in seconds
width = 0.12
amp = 70
noise_std = 1

# Create Gaussian taper
gauss = amp * np.exp(-((times - peak_time) ** 2) / (2 * width**2))

# Initialize all dipole data
data = np.zeros((n_trials, n_chs, n_samples))

for i in range(n_trials):
    dipole_data = np.random.randn(gain.shape[2], n_samples) * noise_std
    # dipole_data[dip_loc, :] = amp * np.sin(2 * np.pi * 10 * times) * gauss
    dipole_data[dip_loc, :] = gauss

    signal = gain[:, 0, :] @ dipole_data
    data[i, :, :] = signal

# Plot the data
plot_EEG(times, data, ch_idx=30)

# %% 4) Non-stationary oscillation in one dipole, transient oscillation in another dipole, noise in all dipoles
# First pick two dipoles
dip_loc1 = 108
dip_loc2 = 509

# Plot brain dipoles
fig = plt.figure()
ax1 = plt.subplot(131, projection='3d')
ax2 = plt.subplot(132)
ax3 = plt.subplot(133)
# Scatter plot the channel positions
ax1.scatter(grid_loc[:, 0], grid_loc[:, 1], grid_loc[:, 2], marker='o', s=8, alpha=0.3)
ax1.scatter(
    grid_loc[dip_loc1, 0],
    grid_loc[dip_loc1, 1],
    grid_loc[dip_loc1, 2],
    marker='o',
    s=50,
    color='red',
)
ax1.scatter(
    grid_loc[dip_loc2, 0],
    grid_loc[dip_loc2, 1],
    grid_loc[dip_loc2, 2],
    marker='o',
    s=50,
    color='orange',
)

ax1.set_title('Brain Dipole Locations')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('z')

# Each dipole can be projected onto the scalp using the forward model.
# The code below shows this projection from one dipole.
im, _ = plot_topomap(
    -gain[:, 0, dip_loc1],
    info,
    names=np.arange(n_chs) + 1,
    axes=ax2,
    cmap=cmap,
    sphere='eeglab',
)
plt.colorbar(im)
im, _ = plot_topomap(
    -gain[:, 0, dip_loc2],
    info,
    names=np.arange(n_chs) + 1,
    axes=ax3,
    cmap=cmap,
    sphere='eeglab',
)
plt.colorbar(im)

ax2.set_title('Signal Dipole Projection')
ax3.set_title('Signal Dipole Projection')

plt.show()

# %%
# Define the parameters
peak_time = 1  # in seconds
width = 0.12
amp = 1
sine_freq = 7
noise_std = 0.2
k = 10

# Create Gaussian taper
gauss = amp * np.exp(-((times - peak_time) ** 2) / (2 * width**2))

# Initialize all dipole data
data = np.zeros((n_trials, n_chs, n_samples))

# Repeat that for N trials
for i in range(n_trials):
    dipole_data = np.random.randn(gain.shape[2], n_samples) * noise_std

    # Non-stationary oscillation in dipole1 (range of 5-10 Hz)
    xi = np.arange(k)
    yi = np.random.rand(k)
    x = np.linspace(0, k - 1, n_samples)
    y = np.interp(x, xi, yi)
    freq_mult = 5 + 5 * y

    dipole_data[dip_loc1, :] = amp * np.sin(
        2 * np.pi * (times + np.cumsum(freq_mult)) / srate
    )

    # Transient oscillation in dipole2
    dipole_data[dip_loc2, :] = amp * np.sin(2 * np.pi * sine_freq * times) * gauss

    # Compute one trial
    signal = gain[:, 0, :] @ dipole_data
    data[i, :, :] = signal

# Plot the data
plot_EEG(times, data, ch_idx=55)
plot_EEG(times, data, ch_idx=30)

# %%
