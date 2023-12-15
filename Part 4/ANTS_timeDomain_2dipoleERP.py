# %%
#     COURSE: Solved problems in neural time series analysis
#    SECTION: Time-domain analyses
#      VIDEO: Simulate ERPs from two dipoles
# Instructor: sincxpress.com

# !%matplotlib inline
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mne import create_info
from mne.channels import make_standard_montage
from mne.viz import plot_topomap
from scipy.io import loadmat
from tqdm import tqdm

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
    freqs_psd = np.linspace(0, sfreq, n_samples)

    # Frequencies in Hz (hard-coded to 2 to 30 in 40 steps)
    freqs = np.linspace(2, 30, 40)
    # Number of wavelet cycles (hard-coded to 3 to 10)
    waves = 2 * (np.linspace(3, 10, len(freqs)) / (2 * np.pi * freqs)) ** 2

    # Setup wavelet and convolution parameters
    wave_t = np.arange(-2, 2 + 1 / sfreq, 1 / sfreq)
    half_w = int(np.floor(len(wave_t) / 2) + 1)
    n_conv = n_samples * n_trials + len(wave_t) - 1

    if data.ndim == 3:
        # ERP
        ax_dict['ERP'].plot(times, data[:, ch_idx, :].T, color='grey', alpha=0.1)
        ax_dict['ERP'].plot(times, np.mean(data[:, ch_idx, :].T, axis=1), color='black')

        # PSD
        # Perform FFT along the columns (n_trials)
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
mat = loadmat(os.path.join('..', 'data', 'emptyEEG.mat'))

grid_loc = mat['lf']['GridLoc'][0][0]
gain = mat['lf']['Gain'][0][0]

sfreq = mat['EEG'][0][0]['srate'][0][0]
n_samples = 2000
times = np.arange(0, n_samples) / sfreq
n_trials = 100
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
n_chs = len(ch_df)

info = create_info(ch_df['label'].to_list(), sfreq, 'eeg')
montage = make_standard_montage('standard_1020')
info.set_montage(montage)

# Initialize channel data
data = np.zeros((n_trials, n_chs, n_samples))

# Pick two dipoles
dip_loc1 = 108
dip_loc2 = 408

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
    color='black',
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
    vlim=(-40, 40),
    show=False,
)

im, _ = plot_topomap(
    -gain[:, 0, dip_loc2],
    info,
    names=np.arange(n_chs) + 1,
    axes=ax3,
    cmap=cmap,
    sphere='eeglab',
    vlim=(-40, 40),
    show=False,
)

# fig.colorbar(im, ax=ax2)
ax2.set_title('First Dipole Projection')
ax3.set_title('Second Dipole Projection')
plt.tight_layout()

plt.show()

# %% Generate dipole -> EEG activity

# Fixed IF signal for dipole1
freq_mod = 2 + 5 * np.interp(
    np.linspace(1, 10, n_samples),
    np.linspace(1, 10, 10),
    np.random.rand(10),
)
if_signal = np.sin(2 * np.pi * ((times + np.cumsum(freq_mod)) / sfreq))

for trial in tqdm(range(n_trials)):
    # Generate dipole activity
    dip_act = 0.02 * np.random.randn(gain.shape[2], n_samples)

    # Dipole 1
    fwhm = np.random.randn() / 10 + 0.3  # Full width at half maximum
    gauss = np.exp(-4 * np.log(2) * (times - 1) ** 2 / fwhm**2)
    dip_act[dip_loc1, :] = if_signal * gauss

    # Dipole 2
    fwhm = np.random.randn() / 10 + 0.3
    gauss = np.exp(-4 * np.log(2) * (times - 1) ** 2 / fwhm**2)
    dip_act[dip_loc2, :] = gauss

    # Now project to the scalp
    data[trial, :, :] = gain[:, 0, :] @ dip_act

# %%
# Plot data from a few channels
plot_EEG(times, data, ch_idx=30)
plot_EEG(times, data, ch_idx=18)
plot_EEG(times, data, ch_idx=10)

# %%
