# %%
#     COURSE: Solved problems in neural time series analysis
#    SECTION: Time-frequency analyses
#    TEACHER: Mike X Cohen, sincxpress.com
#      VIDEO: Baseline normalize power with dB and % change

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

# %% Compare dB and %change in simulated numbers

activity = np.arange(1, 20.01, 0.01)
baseline = 10

# Create normalized vectors
db = 10 * np.log10(activity / baseline)
pc = 100 * (activity - baseline) / baseline

# Plotting
plt.figure()
plt.plot(activity, label='activity')
plt.plot(db, label='dB')
plt.legend()
plt.show()

# Compare dB and percent change directly
plt.figure()
plt.plot(db, pc, 'k')
plt.xlabel('dB')
plt.ylabel('Percent Change')

# Find indices where db is closest to -/+2
db_of_plus2 = np.argmin(np.abs(db - 2))
db_of_minus2 = np.argmin(np.abs(db + 2))

# Plot as guide lines
ax_lim = plt.axis()
plt.plot([db[db_of_plus2], db[db_of_plus2]], [pc[db_of_plus2], ax_lim[2]], 'k', lw=1)
plt.plot([ax_lim[0], db[db_of_plus2]], [pc[db_of_plus2], pc[db_of_plus2]], 'k', lw=1)
plt.plot([db[db_of_minus2], db[db_of_minus2]], [pc[db_of_minus2], ax_lim[2]], 'k', lw=1)
plt.plot([ax_lim[0], db[db_of_minus2]], [pc[db_of_minus2], pc[db_of_minus2]], 'k', lw=1)

plt.show()

# %% Now for real data!

# Load the .mat file
mat = loadmat(os.path.join('..', 'data', 'sampleEEGdata.mat'))
# Get the time points
time = mat['EEG'][0][0][14][0]
# Get the epoched data of shape (channels x samples x trials)
data = mat['EEG'][0][0][15]
# Get the sampling frequency
srate = mat['EEG'][0][0][11][0][0]
# Initialize lists for channel names and coordinates
ch_names = []
# Get the list of channel locations
ch_locs = mat['EEG'][0][0][21][0]
# Iterate over the channels
for ch_loc in ch_locs:
    # Append the channel name
    ch_names.append(ch_loc[0][0])

# Get the number of samples and trials
n_chs, n_pnts, n_trials = data.shape

# Baseline window and convert into indices
base_win = [-500, -200]
[np.argmin(np.abs(time - bw)) for bw in base_win]
base_idx = [np.argmin(np.abs(time - bw)) for bw in base_win]

# %% Setup wavelet parameters

# Frequency parameters
min_freq = 2
max_freq = 30
num_freqs = 40
freqs = np.linspace(min_freq, max_freq, num_freqs)

# Which channel to plot
ch = 'O1'

# Other wavelet parameters
fwhms = np.logspace(np.log10(0.6), np.log10(0.3), num_freqs)
wave_time = np.arange(-2, 2 + 1 / srate, 1 / srate)
half_wave = (len(wave_time) - 1) // 2

# FFT parameters
n_wave = len(wave_time)
n_data = n_pnts * n_trials
n_conv = n_wave + n_data - 1

# Now compute the FFT of all trials concatenated
all_data = data[ch_names.index(ch), :].reshape(-1, order='F')
dataX = np.fft.fft(all_data, n_conv)

# Initialize output time-frequency data
tf = np.zeros((num_freqs, n_pnts))

# %% Now perform convolution

for fi in range(len(freqs)):
    # Create wavelet and get its FFT
    wavelet = np.exp(2 * 1j * np.pi * freqs[fi] * wave_time) * np.exp(
        -4 * np.log(2) * wave_time**2 / fwhms[fi] ** 2
    )
    waveletX = np.fft.fft(wavelet, n_conv)
    waveletX = waveletX / np.max(waveletX)

    # Now run convolution in one step
    comp_sig = np.fft.ifft(waveletX * dataX)
    comp_sig = comp_sig[half_wave:-half_wave]

    # And reshape back to time X trials
    comp_sig = comp_sig.reshape(n_pnts, n_trials, order='F')

    # Compute power and average over trials
    tf[fi, :] = np.mean(np.abs(comp_sig) ** 2, axis=1)

# %% Baseline normalization

# Baseline power
baseline_power = np.mean(tf[:, base_idx[0] : base_idx[1]], axis=1)

# Decibel
tf_db = 10 * np.log10(tf / baseline_power[:, None])

# Percent change
tf_pc = 100 * (tf - baseline_power[:, None]) / baseline_power[:, None]

# %% Plot results

# Define color limits
clim_raw = [0, 2]
clim_db = [-3, 3]
clim_pct = [-90, 90]

plt.figure()

# Define the grid
grid_size = (4, 6)  # Rows, columns
ax1 = plt.subplot2grid(grid_size, (0, 0), colspan=2, rowspan=2)
ax2 = plt.subplot2grid(grid_size, (0, 2), colspan=2, rowspan=2)
ax3 = plt.subplot2grid(grid_size, (0, 4), colspan=2, rowspan=2)
ax4 = plt.subplot2grid(grid_size, (2, 0), colspan=3, rowspan=2)
ax5 = plt.subplot2grid(grid_size, (2, 3), colspan=3, rowspan=2)

# Raw power
c = ax1.contourf(time, freqs, tf, 40, cmap=cmap)
c.set_clim(clim_raw)
ax1.set_xlim([-300, 1000])
ax1.set_xlabel('Time (ms)')
ax1.set_ylabel('Frequency (Hz)')
ax1.set_title('Raw Power')

# dB power
c = ax2.contourf(time, freqs, tf_db, 40, cmap=cmap)
c.set_clim(clim_db)
ax2.set_xlim([-300, 1000])
ax2.set_xlabel('Time (ms)')
ax2.set_title('dB Power')

# Percent change power
c = ax3.contourf(time, freqs, tf_pc, 40, cmap=cmap)
c.set_clim(clim_pct)
ax3.set_xlim([-300, 1000])
ax3.set_xlabel('Time (ms)')
ax3.set_title('Pct Change Power')

# Raw power vs. pct change power
ax4.plot(tf.flatten(), tf_pc.flatten(), 'rs', markerfacecolor='k')
ax4.set_xlabel('Raw')
ax4.set_ylabel('Pct Change')
ax4.set_title('Raw Power vs. Pct Change Power')

# Pct change power vs. dB power
ax5.plot(tf_db.flatten(), tf_pc.flatten(), 'rs', markerfacecolor='k')
ax5.set_xlabel('dB')
ax5.set_ylabel('% $\Delta$')
ax5.set_title('Pct Change Power vs. dB Power')

plt.tight_layout()
plt.show()

# %%
