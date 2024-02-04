# %%
#   COURSE: Neural signal processing and analysis: Zero to hero
#  SECTION: Time-frequency analysis
#  TEACHER: Mike X Cohen, sincxpress.com
#    VIDEO: Within-subject, cross-trial regression

# !%matplotlib inline
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from scipy.fftpack import fft, ifft
from scipy.io import loadmat

# Set figure settings
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import get_cmap, set_fig_dpi, set_style

set_fig_dpi(), set_style()
cmap = get_cmap('parula')
# %%
# Note about this dataset:
# data is frequencies X trials
# The goal is to test whether EEG frequency power is related to RT over trials
# Load data
# Assuming the data is loaded into Python in a compatible format
mat = loadmat(os.path.join('..', 'data', 'EEG_RT_data.mat'))
rts = mat['rts'][0]
data = mat['EEGdata']
freqs = mat['frex'][0]
N = len(rts)

# Show the data
plt.figure()
plt.subplot(211)
plt.plot(rts, 'ks-', markersize=7)
plt.ylabel('Response Time (ms)')

plt.subplot(212)
plt.imshow(data, origin='lower', aspect='auto', cmap=cmap)
plt.clim([0, 10])
plt.xlabel('Trial')
plt.ylabel('Frequency (Hz)')

plt.tight_layout()
plt.show()

# Q: Is there "a lot" or "a little" variability in RT or brain over trials?
# A: Yes, the variability in reaction times is quite large
#    and the variability in EEG power is also quite large.
# %% Compute effect over frequencies
b = np.zeros(len(freqs))
for fi in range(len(freqs)):
    X = np.vstack([np.ones(N), data[fi, :]]).T
    t = np.linalg.lstsq(X, rts, rcond=None)[0]
    b[fi] = t[1] * np.std(data[fi, :])

# Plot
plt.figure()
plt.subplot(2, 1, 1)
plt.plot(freqs, b, 'rs-', markersize=7, markerfacecolor='k')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Beta-Coefficient')

# scatterplots at these frequencies
freq_idx = [np.argmin(np.abs(freqs - freq)) for freq in [8, 20]]

for fi in range(2):
    plt.subplot(2, 2, 3 + fi)
    plt.scatter(data[freq_idx[fi], :], rts, s=50, facecolors='k')
    plt.title(f'EEG signal at {[8,20][fi]} Hz')
    plt.xlabel('EEG Energy')
    plt.ylabel('RT')

    # Least square fit
    X = np.vstack([np.ones(N), data[freq_idx[fi], :]]).T
    t = np.linalg.lstsq(X, rts, rcond=None)[0]
    fit = X @ t

    plt.plot(data[freq_idx[fi], :], fit, 'r-', linewidth=2)

plt.tight_layout()
plt.show()

# %% Load EEG data and extract reaction times in ms
# Load the .mat file
mat = loadmat(os.path.join('..', 'data', 'sampleEEGdata.mat'))
# Get the time points
times = mat['EEG'][0][0][14][0]
# Get the epoched data of shape (channels x samples x trials)
epoched_data = mat['EEG'][0][0][15]
# Get the sampling frequency
sfreq = mat['EEG'][0][0][11][0][0]
# Get the event latencies
event_lats = mat['EEG']['epoch'][0, 0]['eventlatency'][0]

_, n_pnts, n_trials = epoched_data.shape

# Extracts the reaction time from each trial
rts = np.zeros((n_trials))

# Loop over trials
for ei in range(n_trials):
    # Find the index corresponding to time=0, i.e., trial onset
    zero_loc = np.argmin(np.abs(event_lats[ei]))

    # Reaction time is the event after the trial onset
    rts[ei] = event_lats[ei][0, zero_loc + 1][0, 0]

# Always good to inspect data, check for outliers, etc.
plt.figure()
plt.plot(rts, 'o-', markerfacecolor='w', markersize=7)
plt.xlabel('Trial')
plt.ylabel('Reaction Time (ms)')
plt.show()

# %% Create the design matrix
# Our design matrix will have two regressors (two columns): intercept and RTs
X = np.column_stack([np.ones(n_trials), rts])

# %% Run wavelet convolution for time-frequency analysis
# We didn't cover this in class, but this code extracts a time-frequency
# map of power for each trial. These power values become the dependent variables.

freq_range = [2, 25]  # Extract only these frequencies (in Hz)
n_freqs = 30  # Number of frequencies between lowest and highest

# Set up convolution parameters
wave_time = np.arange(-2, 2 + 1 / sfreq, 1 / sfreq)
freqs = np.linspace(freq_range[0], freq_range[1], n_freqs)
n_data = n_pnts * n_trials
n_kern = len(wave_time)
n_conv = n_data + n_kern - 1
half_wave = (len(wave_time) - 1) // 2
n_cycles = np.logspace(np.log10(4), np.log10(12), n_freqs)

# Initialize time-frequency output matrix
tf_3d = np.zeros((n_freqs, n_pnts, n_trials))

# Compute Fourier coefficients of EEG data (doesn't change over frequency!)
eegX = fft(epoched_data[46, :, :].T.reshape(-1), n_conv)

# Loop over frequencies
for fi in range(n_freqs):
    # Create the wavelet
    s = n_cycles[fi] / (2 * np.pi * freqs[fi])
    cmw = np.exp(2 * 1j * np.pi * freqs[fi] * wave_time) * np.exp(
        (-(wave_time**2)) / (2 * s**2)
    )
    cmwX = fft(cmw, n_conv)
    cmwX /= np.max(cmwX)

    # Second and third steps of convolution
    comp_sig = ifft(eegX * cmwX)

    # Cut wavelet back to size of data
    comp_sig = comp_sig[half_wave:-half_wave]
    comp_sig = comp_sig.reshape(n_trials, n_pnts).T

    # Extract power from all trials
    tf_3d[fi, :, :] = np.abs(comp_sig) ** 2

# %% Inspect the TF plots a bit
plt.figure()

# Show the raw power maps for three trials
for i in range(3):
    plt.subplot(2, 3, i + 1)
    plt.imshow(
        tf_3d[:, :, i],
        aspect='auto',
        extent=[times[0], times[-1], freqs[0], freqs[-1]],
        origin='lower',
        cmap=cmap,
    )
    plt.clim([0, 10])
    plt.xlim([-200, 1200])
    plt.xlabel('Time (ms)')
    plt.ylabel('Frequency')
    plt.title(f'Trial {i + 1}')

# Now show the trial-average power map
plt.subplot(212)
plt.imshow(
    np.mean(tf_3d, axis=2),
    aspect='auto',
    extent=[times[0], times[-1], freqs[0], freqs[-1]],
    origin='lower',
    cmap=cmap,
)
plt.clim([0, 5])
plt.xlim([-200, 1200])
plt.xlabel('Time (ms)')
plt.ylabel('Frequency')
plt.title('Average Over All Trials')

plt.tight_layout()
plt.show()

# %% Now for the regression model
# We're going to take a short-cut here, and reshape the 3D matrix to 2D.
# That doesn't change the values, and we don't alter the trial order.
# Note the size of the matrix below.
tf_2d = tf_3d.reshape(n_freqs * n_pnts, n_trials, order='F').T

# Now we can fit the model on the 2D matrix
b = np.linalg.pinv(X.T @ X) @ (X.T @ tf_2d)

# Reshape b into a time-by-frequency matrix
beta_mat = b[1, :].reshape(n_pnts, n_freqs).T

# %% Show the design and data matrices
fig = plt.figure()
gs = GridSpec(1, 8, figure=fig)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1:])

# Design Matrix
ax1.imshow(X, aspect='auto', cmap='gray', origin='lower')
ax1.set_title('Design Matrix')
ax1.set_ylabel('Trials')
ax1.set_xticks([0, 1])
ax1.set_xticklabels(['Int', 'RT'])

# Data Matrix
ax2.imshow(
    tf_2d,
    clim=[0, 20],
    origin='lower',
    aspect='auto',
    cmap='gray',
    interpolation='none',
)
ax2.set_title('Data Matrix')
ax2.set_xlabel('Time-Frequency')
ax2.set_yticks([])

plt.tight_layout()
plt.show()

# Q: Please interpret the matrices.
#    What do they mean and what do they show?
# A: The design matrix shows the two regressors (intercept and RT).
#    The data matrix shows the power at each time-frequency point.

# %% Show the results
plt.figure()

# Show time-frequency map of regressors
plt.contourf(times, freqs, beta_mat, 40, cmap=cmap)
plt.title('Regression against RT over trials')
plt.xlabel('Time (ms)')
plt.ylabel('Frequency (Hz)')
plt.xlim(-200, 1200)
plt.clim(-0.012, 0.012)
plt.show()

# Q: How do you interpret the results?
# A: The yellow blob shows that RT is positively correlated with theta power
#    between 600-800 ms. The blue blob shows that RT is negatively correlated
#    with alpha between 200-400 ms.
# Q: Do you believe these results? Are they statistically significant?
# A: Apparently, the yellow blob is statistically significant, but the blue blob is not.

# %%
