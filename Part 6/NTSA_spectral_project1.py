# %%
#   COURSE: Solved challenges in neural time series analysis
#  SESSION: Spectral analyses
#  TEACHER: Mike X Cohen, sincxpress.com
#    VIDEO: Project 3-1: Topography of spectrally separated activity
#     GOAL: Separate two sources based on power spectra.
#           Plot topographies and compute the fit of the data to the templates via R^2.

# !%matplotlib inline
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from mne import create_info
from mne.channels import make_standard_montage
from mne.viz import plot_topomap
from scipy.io import loadmat

# Set figure settings
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import get_cmap, set_fig_dpi, set_style

set_fig_dpi(), set_style(notebook=True)
cmap = get_cmap('parula')


# %%
def plot_EEG(times, data, ch_idx=0):
    times = times.copy() * 1000
    n_samples = len(times)
    mosaic = [['ERP', 'ERP'], ['PSD', 'TF']]
    fig = plt.figure(constrained_layout=True)
    ax_dict = fig.subplot_mosaic(mosaic)

    ax_dict['ERP'].plot(times, data[ch_idx, :, :], color='grey', alpha=0.1)
    ax_dict['ERP'].plot(times, np.mean(data[ch_idx, :, :], axis=1), color='black')
    ax_dict['ERP'].set_title(f'ERP From Channel {ch_idx+1}')
    ax_dict['ERP'].set_xlabel('Time (ms)')
    ax_dict['ERP'].set_ylabel('Activity')
    ax_dict['ERP'].set_xlim(times.min(), times.max())

    freqs = np.linspace(0, srate, n_samples)
    if data.ndim == 3:
        # Perform FFT along the columns (trials)
        pw = np.mean(
            (2 * np.abs(np.fft.fft(data[ch_idx, :, :], axis=0) / n_samples)) ** 2,
            axis=1,
        )
    else:
        pw = (2 * np.abs(np.fft.fft(data[ch_idx, :], axis=0) / n_samples)) ** 2
    ax_dict['PSD'].plot(freqs, pw)
    ax_dict['PSD'].set_title(f'Static Power Spectrum')
    ax_dict['PSD'].set_xlabel('Frequency (Hz)')
    ax_dict['PSD'].set_ylabel('Power')
    ax_dict['PSD'].set_xlim(0, 40)

    # Frequencies in Hz (hard-coded to 2 to 30 in 40 steps)
    freqs = np.linspace(2, 30, 40)
    # Number of wavelet cycles (hard-coded to 3 to 10)
    waves = 2 * (np.linspace(3, 10, len(freqs)) / (2 * np.pi * freqs)) ** 2

    # Setup wavelet and convolution parameters
    wave_t = np.arange(-2, 2 + 1 / srate, 1 / srate)
    half_w = int(np.floor(len(wave_t) / 2) + 1)
    n_conv = n_samples * n_trials + len(wave_t) - 1

    # Initialize the time-frequency matrix
    tf_mat = np.zeros((len(freqs), n_samples))

    # Spectrum of data
    data_fft = np.fft.fft(data[ch_idx, :, :].reshape(1, -1, order='F'), n_conv)

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

    ax_dict['TF'].contourf(times, freqs, tf_mat, 40, cmap='jet')
    ax_dict['TF'].set_title(f'Time-Frequency Plot')
    ax_dict['TF'].set_xlabel('Time (ms)')
    ax_dict['TF'].set_ylabel('Frequency (Hz)')
    ax_dict['TF'].set_xlim(times.min(), times.max())

    plt.show()


# Load the mat file containing EEG, leadfield and channel locations
mat = loadmat(os.path.join('..', 'data', 'emptyEEG'))
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
srate = 1024

gain = mat['lf'][0][0][2]
grid_loc = mat['lf'][0][0][5]

info = create_info(ch_df['label'].to_list(), srate, 'eeg')
montage = make_standard_montage('standard_1020')
info.set_montage(montage)

# %%
# Select dipole location
dip_loc1 = 108
dip_loc2 = 117

# Plot brain dipoles
fig = plt.figure()

gs = GridSpec(2, 2, figure=fig)
ax1 = fig.add_subplot(gs[0:2, 0], projection='3d')
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, 1])

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
    color='black',
)

ax1.set_title('Brain Dipole Locations')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('z')

# Each dipole can be projected onto the scalp using the forward model
im, _ = plot_topomap(
    -gain[:, 0, dip_loc1],
    info,
    names=np.arange(n_chs) + 1,
    axes=ax2,
    cmap=cmap,
    vlim=[-40, 40],
    sphere='eeglab',
    show=False,
)
ax2.set_title('First Dipole Projection')

im, _ = plot_topomap(
    -gain[:, 0, dip_loc2],
    info,
    names=np.arange(n_chs) + 1,
    axes=ax3,
    cmap=cmap,
    vlim=[-40, 40],
    sphere='eeglab',
    show=False,
)
ax3.set_title('Second Dipole Projection')

plt.tight_layout()
plt.show()

# %% Adjust EEG parameters
n_pnts = 1143
n_trials = 150
n_chan = 64
time = np.arange(n_pnts) / srate - 0.2

# Initialize EEG data
data = np.zeros((n_chan, n_pnts, n_trials))

# Create simulated data

# Gaussian
peak_time = 0.5  # seconds
fwhm = 0.12

# Create Gaussian taper
gauss = np.exp(-(4 * np.log(2) * (time - peak_time) ** 2) / fwhm**2)

sine_freq1 = 9
sine_freq2 = 14

sine1 = np.sin(2 * np.pi * sine_freq1 * time)
sine2 = np.sin(2 * np.pi * sine_freq2 * time)

for trial in range(n_trials):
    # Initialize all dipole data
    dip_data = 0.01 * np.random.randn(gain.shape[2], n_pnts)

    dip_data[dip_loc1, :] = sine1 * gauss
    dip_data[dip_loc2, :] = sine2 * gauss

    # Compute one trial
    data[:, :, trial] = gain[:, 0, :] @ dip_data

# %% Try a few channels
plot_EEG(time, data, 19)
plot_EEG(time, data, 29)
plot_EEG(time, data, 28)

# %% Now for the project...

# FFT of all channels
pwr = np.mean((2 * np.abs(np.fft.fft(data, axis=1) / n_pnts)) ** 2, axis=2)

# Vector of frequencies
hz = np.linspace(0, srate / 2, int(np.floor(n_pnts / 2) + 1))

# Frequency cutoffs in Hz and indices
cutoffs1 = np.array([2, 11]).reshape(-1, 1)
cutoffs2 = np.array([11, 20]).reshape(-1, 1)

indices1 = np.squeeze(np.argmin(np.abs(hz - cutoffs1), axis=1))
indices2 = np.squeeze(np.argmin(np.abs(hz - cutoffs2), axis=1))

win1 = np.arange(indices1[0], indices1[1] + 1)
win2 = np.arange(indices2[0], indices2[1] + 1)

# Power in first spectral window
avg_pwr_win1 = pwr[:, win1].mean(axis=1)
avg_pwr_win2 = pwr[:, win2].mean(axis=1)

# %%
# Topographical plots of dipole projections and band-specific power.
fig = plt.figure()

gs = GridSpec(2, 2, figure=fig)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, 0])
ax4 = fig.add_subplot(gs[1, 1])

# Dipole projections
im, _ = plot_topomap(
    -gain[:, 0, dip_loc1],
    info,
    names=np.arange(n_chs) + 1,
    axes=ax1,
    cmap=cmap,
    vlim=[-40, 40],
    sphere='eeglab',
    show=False,
)
ax1.set_title('First Dipole Projection')

im, _ = plot_topomap(
    -gain[:, 0, dip_loc2],
    info,
    names=np.arange(n_chs) + 1,
    axes=ax2,
    cmap=cmap,
    vlim=[-40, 40],
    sphere='eeglab',
    show=False,
)
ax2.set_title('Second Dipole Projection')

# Band-specific power
im, _ = plot_topomap(
    avg_pwr_win1,
    info,
    names=np.arange(n_chs) + 1,
    axes=ax3,
    cmap=cmap,
    sphere='eeglab',
    show=False,
)
ax3.set_title(f'Average Power in\nSpectral Window ({cutoffs1[0,0]}-{cutoffs1[1,0]} Hz)')

im, _ = plot_topomap(
    avg_pwr_win2,
    info,
    names=np.arange(n_chs) + 1,
    axes=ax4,
    cmap=cmap,
    sphere='eeglab',
    show=False,
)
ax4.set_title(f'Average Power in\nSpectral Window ({cutoffs2[0,0]}-{cutoffs2[1,0]} Hz)')

plt.tight_layout()
plt.show()

# %%
# Quantify fit via R2

# Data to each dipole projection
low_to_dip1 = np.corrcoef(avg_pwr_win1, gain[:, 0, dip_loc1])[0, 1] ** 2
low_to_dip2 = np.corrcoef(avg_pwr_win1, gain[:, 0, dip_loc2])[0, 1] ** 2
high_to_dip1 = np.corrcoef(avg_pwr_win2, gain[:, 0, dip_loc1])[0, 1] ** 2
high_to_dip2 = np.corrcoef(avg_pwr_win2, gain[:, 0, dip_loc2])[0, 1] ** 2

# Bar plot of fits
plt.figure()
plt.bar([1, 2, 4, 5], [low_to_dip1, low_to_dip2, high_to_dip1, high_to_dip2])

plt.ylabel('Fit to Ground Truth ($R^2$)')
plt.xticks(
    [1, 2, 4, 5],
    ['Low-Dipole 1', 'Low-Dipole 2', 'High-Dipole 1', 'High-Dipole 2'],
    rotation=45,
)
plt.show()

# %%
