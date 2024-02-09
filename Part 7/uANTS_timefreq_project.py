# %%
#   COURSE: Neural signal processing and analysis: Zero to hero
#  SECTION: Time-frequency analysis
#  TEACHER: Mike X Cohen, sincxpress.com
#  SESSION: Time-frequency analysis: Problem set

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from scipy.signal import filtfilt, firwin, hilbert

# !%matplotlib inline
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import get_cmap, set_fig_dpi, set_style

# Set figure settings
set_fig_dpi(), set_style()
cmap = get_cmap('parula')
# %% 1) Power and phase from the famous "trial 10"
# Create a family of complex Morlet wavelets that range from 10 Hz to 100 Hz in 43
# linearly spaced steps. Perform convolution between the wavelets and V1 data from
# trial 10 for all channels. Extract power and phase, store the results in a
# channel X frequency X time X pow/phs (thus, 4D) matrix.

# Load the .mat file
mat = loadmat(os.path.join('..', 'data', 'v1_laminar.mat'))
# Get the time points
timevec = mat['timevec'][0]
# Get the epoched data of shape (channels x samples x trials)
data = mat['csd']
n_chs, n_samples, n_trials = data.shape
# Get the sampling frequency
srate = mat['srate'][0][0]


#  Soft-coded parameters
freq_range = [10, 100]  # Extract only these frequencies (in Hz)
n_freqs = 43  # Number of frequencies between lowest and highest
trial_num = 9

# Set up convolution parameters
wave_time = np.arange(-2, 2 - 1 / srate, 1 / srate)
freqs = np.linspace(freq_range[0], freq_range[1], n_freqs)
n_data = len(timevec)
n_kern = len(wave_time)
n_conv = n_data + n_kern - 1
half_wave = (len(wave_time) - 1) // 2

# Number of cycles
n_cyc = np.linspace(3, 15, n_freqs)

# Create wavelets
cmwX = np.zeros((n_freqs, n_conv), dtype=complex)
for fi in range(n_freqs):
    # Create time-domain wavelet
    two_s_squared = 2 * (n_cyc[fi] / (2 * np.pi * freqs[fi])) ** 2
    cmw = np.exp(2 * 1j * np.pi * freqs[fi] * wave_time) * np.exp(
        (-(wave_time**2)) / two_s_squared
    )

    # Compute Fourier coefficients of wavelet and normalize
    cmwX[fi, :] = np.fft.fft(cmw, n_conv)
    cmwX[fi, :] /= cmwX[fi, :][np.argmax(np.abs(cmwX[fi, :]))]

# Initialize time-frequency output matrix
tf = np.zeros((n_chs, n_freqs, len(timevec), 2))

# Loop over channels
for chani in range(n_chs):
    # Compute Fourier coefficients of EEG data (doesn't change over frequency!)
    eegX = np.fft.fft(data[chani, :, trial_num], n_conv)

    # Loop over frequencies
    for fi in range(n_freqs):
        # Second and third steps of convolution
        comp_sig = np.fft.ifft(cmwX[fi, :] * eegX, n_conv)

        # Cut wavelet back to size of data
        comp_sig = comp_sig[half_wave:-half_wave]

        # Extract power and phase
        tf[chani, fi, :, 0] = np.abs(comp_sig) ** 2
        tf[chani, fi, :, 1] = np.angle(comp_sig)


# %% Plotting the results, part 1: time-frequency
# In a 1x2 subplot figure, plot time-frequency power (left) and phase
# (right) from electrode 6. Use x-axis scaling of -200 to +1000 ms.
ch = 5

plt.figure()
# Plot power
plt.subplot(121)
plt.contourf(timevec, freqs, tf[ch, :, :, 0], 40, cmap=cmap)
plt.title(f'Power from Trial 10 at Contact {ch+1}')
plt.xlabel('Time (s)')
plt.ylabel('Frequencies (Hz)')
plt.xlim([-0.2, 1])
plt.clim([0, 80000])

# Plot phase
plt.subplot(122)
plt.contourf(timevec, freqs, tf[ch, :, :, 1], 40, cmap=cmap)

plt.xlabel('Time (s)')
plt.ylabel('Frequencies (Hz)')
plt.title(f'Phase from Trial 10 at Contact {ch+1}')
plt.xlim([-0.2, 1])
plt.clim([-np.pi, np.pi])

plt.tight_layout()
plt.show()

# Q: Can you use the same color scaling for the two plots? Why or why not?
# A: No, because the power and phase values are on different scales.

# %% Plotting the results, part 2: Depth-by-time
# Make four layer-by-time maps in a 2x2 subplot figure. Plot power (top row)
# and phase (bottom row), from data at 40 Hz and at 55 Hz. Are there differences
# between power and phase, and would you expect to see differences or similarities?

hz_idx = [np.argmin(np.abs(freqs - 40)), np.argmin(np.abs(freqs - 55))]

fig, axs = plt.subplots(2, 2)

c = axs[0, 0].contourf(
    timevec, np.arange(1, n_chs + 1), tf[:, hz_idx[0], :, 0], 40, cmap=cmap
)
axs[0, 0].set_title('Power at 40 Hz')
axs[0, 0].set_ylabel('Depth (electrode)')
c.set_clim([0, 40000])

c = axs[0, 1].contourf(
    timevec, np.arange(1, n_chs + 1), tf[:, hz_idx[1], :, 0], 40, cmap=cmap
)
axs[0, 1].set_title('Power at 55 Hz')
c.set_clim([0, 40000])

c = axs[1, 0].contourf(
    timevec, np.arange(1, n_chs + 1), tf[:, hz_idx[0], :, 1], 40, cmap=cmap
)
axs[1, 0].set_title('Phase at 40 Hz')
axs[1, 0].set_xlabel('Time (s)')
axs[1, 0].set_ylabel('Depth (electrode)')
c.set_clim([-np.pi, np.pi])

c = axs[1, 1].contourf(
    timevec, np.arange(1, n_chs + 1), tf[:, hz_idx[1], :, 1], 40, cmap=cmap
)
axs[1, 1].set_title('Phase at 55 Hz')
axs[1, 1].set_xlabel('Time (s)')
c.set_clim([-np.pi, np.pi])

plt.tight_layout()
plt.show()

# Q: How can you interpret the phase plots?
# A: They are quite difficult to interpret. The syncronization within
#    one channel or across multiple channels would be more interesting.
# Q: How do you interpret the two power plots?
# A: The first stimulus induces a wide response at 40 Hz.
#    The second stimulus induces a narrower response at 55 Hz.


# %% 2) Convolution with all trials
#    Repeat the previous exercise, but using data from all trials.
#    Don't save the single-trial data.
#    Instead of the raw phases, compute ITPC.
#    Generate the same plots as in #2.

# Q: Which parameters/variables do you need to recompute,
#    and which can you reuse from above?

# Set up convolution parameters
n_data = len(timevec) * n_trials
n_conv = n_data + n_kern - 1

# Create wavelets
cmwX = np.zeros((n_freqs, n_conv), dtype=complex)
for fi in range(n_freqs):
    # Create time-domain wavelet
    two_s_squared = 2 * (n_cyc[fi] / (2 * np.pi * freqs[fi])) ** 2
    cmw = np.exp(2 * 1j * np.pi * freqs[fi] * wave_time) * np.exp(
        (-(wave_time**2)) / two_s_squared
    )

    # Compute Fourier coefficients of wavelet and normalize
    cmwX[fi, :] = np.fft.fft(cmw, n_conv)
    cmwX[fi, :] /= np.max(cmwX[fi, :])

# Initialize time-frequency output matrix
tf = np.zeros((n_chs, n_freqs, len(timevec), 2))

# Note about the code below:
# I solved this using no loop over channels, by taking advantage of
# matrix input into the FFT function. I don't generally recommed this
# method, because it can get confusing and you might run into memory
# limitations for large datasets. But it's useful to see how it can work.


# Compute Fourier coefficients of EEG data (doesn't change over frequency!)
eegX = np.fft.fft(data.reshape(n_chs, -1, order='F'), n_conv, axis=1)

# Loop over frequencies
for fi in range(n_freqs):
    # Second and third steps of convolution
    cmwX_rep = np.tile(cmwX[fi, :], (n_chs, 1))
    comp_sig = np.fft.ifft(eegX * cmwX_rep, n_conv, axis=1)

    # Cut wavelet back to size of data
    comp_sig = comp_sig[:, half_wave:-half_wave]

    # Reshape back to original data size [channels, time points, trials]
    comp_sig = comp_sig.reshape(data.shape, order='F')

    # Extract power and phase
    tf[:, fi, :, 0] = np.mean(np.abs(comp_sig) ** 2, axis=2)  # Power
    tf[:, fi, :, 1] = np.abs(np.mean(np.exp(1j * np.angle(comp_sig)), axis=2))  # ITPC


# %%
plt.figure()

# Power plot
plt.subplot(121)
plt.contourf(timevec, freqs, tf[ch, :, :, 0], 40, cmap=cmap)
plt.xlabel('Time (s)')
plt.ylabel('Frequencies (Hz)')
plt.title(f'Power from All Trials at Contact {ch+1}')
plt.xlim([-0.2, 1])
plt.clim([0, 80000])

# Phase (ITPC) plot
plt.subplot(122)
plt.contourf(timevec, freqs, tf[ch, :, :, 1], 40, cmap=cmap)
plt.xlabel('Time (s)')
plt.title(f'ITPC from All Trials at Contact {ch+1}')
plt.xlim([-0.2, 1])
plt.clim([0, 0.5])

plt.tight_layout()
plt.show()

# %%
# Plotting power and phase at specific frequencies across all channels
plt.figure()

for i in range(2):
    # Power at specific frequencies
    plt.subplot(2, 2, i + 1)
    plt.contourf(
        timevec,
        np.arange(1, tf.shape[0] + 1),
        tf[:, hz_idx[i], :, 0],
        40,
        cmap=cmap,
    )
    plt.title(f'Power at {round(freqs[hz_idx[i]])} Hz')
    plt.clim([0, 40000])
    if i == 0:
        plt.ylabel('Electrode Depth')

    # Phase (ITPC) at specific frequencies
    plt.subplot(2, 2, i + 3)
    plt.contourf(
        timevec,
        np.arange(1, tf.shape[0] + 1),
        tf[:, hz_idx[i], :, 1],
        40,
        cmap=cmap,
    )
    plt.title(f'Phase at {round(freqs[hz_idx[i]])} Hz')
    plt.clim([0, 0.2])
    plt.xlabel('Time (s)')
    if i == 0:
        plt.ylabel('Electrode Depth')

plt.tight_layout()
plt.show()


# %% 3) Exploring edge effects
# Create a square-wave time series and perform a time-frequency
# analysis, in order to explore the effects of edges on TF responses.

sfreq = 999
n_pnts = sfreq * 1
time = np.arange(0, n_pnts) / sfreq

# Create the square wave function
squarets = np.zeros(n_pnts)
squarets[int(np.round(n_pnts * 0.4)) : int(np.round(n_pnts * 0.6))] = 1

# Time-frequency analysis

# Soft-coded parameters
freq_range = [1, 100]  # Extract only these frequencies (in Hz)
n_freqs = 83  # Number of frequencies between lowest and highest


# Set up convolution parameters
wave_time = np.arange(-2, 2 + 1 / sfreq, 1 / sfreq)
freqs = np.linspace(freq_range[0], freq_range[1], n_freqs)
n_data = n_pnts
n_kern = len(wave_time)
n_conv = n_data + n_kern - 1
half_wave = (len(wave_time) - 1) // 2

# Number of cycles
n_cyc = np.linspace(3, 15, n_freqs)


# Compute Fourier coefficients of signal
impfunX = np.fft.fft(squarets, n_conv)


# Initialize TF matrix
tf = np.zeros((n_freqs, n_pnts, 2))


# Create wavelets and do TF decomposition in one loop
for fi in range(n_freqs):
    # Create time-domain wavelet
    two_s_squared = 2 * (n_cyc[fi] / (2 * np.pi * freqs[fi])) ** 2
    cmw = np.exp(2 * 1j * np.pi * freqs[fi] * wave_time) * np.exp(
        (-(wave_time**2)) / two_s_squared
    )

    # Compute fourier coefficients of wavelet and normalize
    cmwX = np.fft.fft(cmw, n_conv)

    # Second and third steps of convolution
    # (The max value of a complex array is defined differently in MATLAB and Python)
    comp_sig = np.fft.ifft(cmwX * impfunX / cmwX[np.argmax(np.abs(cmwX))])

    # Cut wavelet back to size of data
    comp_sig = comp_sig[half_wave:-half_wave]

    # Extract power and phase
    tf[fi, :, 0] = np.abs(comp_sig) ** 2
    tf[fi, :, 1] = np.angle(comp_sig)


# Plot the results
plt.figure()
plt.subplot(311)
plt.plot(time, squarets)
plt.title('Box-Car Time Series')
plt.ylabel('Amplitude (a.u.)')

# Time-frequency power
plt.subplot(312)
plt.imshow(
    tf[:, :, 0],
    aspect='auto',
    extent=[time[0], time[-1], freqs[0], freqs[-1]],
    origin='lower',
    cmap=cmap,
)
plt.clim([0, 0.001])
plt.title('Time-Frequency Power')
plt.ylabel('Frequency (Hz)')

# Time-frequency phase
plt.subplot(313)
plt.contourf(time, freqs, tf[:, :, 1], 40, cmap=cmap)
plt.ylabel('Frequency (Hz)')
plt.xlabel('Time (s)')
plt.title('Time-Frequency Phase')

plt.tight_layout()
plt.show()


# Q: How "bad" is it?
#    Does the edge more adversely affect power or phase?
# A: Depends on the magnitude of the signal.
#    Kind of difficult to answer. They are both affected.
#    The color limits also play a role in this comparison.
# Q: What would you consider a reasonable "buffer" in terms of
#    cycles per frequency to avoid the edge effects?
# A: Three cycles is a good rule of thumb.
# Q: Does the size of the edge effect depend on the amplitude of the box?
# A: Yes, the larger the amplitude, the larger the edge effect.
# Q: Does it also depend on the number of cycles for the wavelet?
# A: Yes, it does.


# %% 4) Improving the spectral precision of wavelet convolution.

# Remember from the first section of the course that we identified a "failure scenario"
# in which wavelet convolution failed to identify two sine waves that were simulated and
# clearly visible in the static spectrum. Let's revisit that example.

sfreq = 300
time = np.arange(0, sfreq * 2) / sfreq

# Create the signal with 4 and 6 Hz components
signal = np.sin(2 * np.pi * 4 * time) + np.sin(2 * np.pi * 6 * time)

# Compute static power spectrum
pwr = np.abs(np.fft.fft(signal) / len(time)) ** 2
hz = np.linspace(0, sfreq, len(time))


# Time-frequency analysis

# Soft-coded parameters
freq_range = [2, 12]  # Extract only these frequencies (in Hz)
n_freqs = 20  # Number of frequencies between lowest and highest


# Set up convolution parameters
wave_time = np.arange(-2, 2 + 1 / sfreq, 1 / sfreq)
freqs = np.linspace(freq_range[0], freq_range[1], n_freqs)
n_data = len(time)
n_kern = len(wave_time)
n_conv = n_data + n_kern - 1
half_wave = (len(wave_time) - 1) // 2

# Full-width half-maximum of wavelets
fwhms = np.linspace(0.5, 0.3, n_freqs)
# fwhms = np.linspace(0.7, 0.7, n_freqs)  # Better results


# Compute Fourier coefficients of signal
impfunX = np.fft.fft(signal, n_conv)


# Initialize TF matrix
tf = np.zeros((n_freqs, n_data))


# Create wavelets and do TF decomposition in one loop
for fi in range(n_freqs):
    # Create time-domain wavelet
    cmw = np.exp(2 * 1j * np.pi * freqs[fi] * wave_time) * np.exp(
        -4 * np.log(2) * (wave_time**2) / fwhms[fi] ** 2
    )

    # Compute fourier coefficients of wavelet and normalize
    cmwX = np.fft.fft(cmw, n_conv)

    # Second and third steps of convolution
    # (The max value of a complex array is defined differently in MATLAB and Python)
    comp_sig = np.fft.ifft(cmwX * impfunX / cmwX.max())

    # Cut wavelet back to size of data
    comp_sig = comp_sig[half_wave:-half_wave]

    # Extract power and phase
    tf[fi, :] = np.abs(comp_sig) ** 2


# Plot the results
plt.figure()
plt.subplot(311)
plt.plot(time, signal)
plt.title('Signal')
plt.ylabel('Amplitude (a.u.)')

# Time-frequency power
plt.subplot(312)
plt.imshow(
    tf,
    aspect='auto',
    extent=[time[0], time[-1], freqs[0], freqs[-1]],
    origin='lower',
    cmap=cmap,
)
plt.clim([0, 0.2])
plt.title('Time-Frequency Power')
plt.xlabel('Times (s)')
plt.ylabel('Frequency (Hz)')

# Time-frequency phase
plt.subplot(313)
plt.stem(hz, pwr)
plt.title('Static Power')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude (a.u.)')
plt.xlim(0, freq_range[1])

plt.tight_layout()
plt.show()

# Q: Is time-frequency analysis completely utterly worthless?!!?
# A: No, one can see the 4 and 6 Hz components and a 2 Hz component
#    which corresponds to their differences.
# Q: Try changing the FWHM limits to see if you can recover the
#    signals. Can you get it better? Can you get it perfect?
# A: Yes setting the FWHM higher decreases time resolution while
#   increasing frequency resolution. So, the frequency components
#   are clearly distinguishable with a higher FWHM.

# %% 5) Compare complex wavelet convolution with filter-Hilbert

# The goal here is to illustrate that complex Morlet wavelet convolution
# can give the same or different results as filter-Hilbert, depending on parameters.

# Compute the ERP from channel 7 in the v1 dataset
erp = np.mean(data[6, :, :], axis=1)

# Wavelet convolution
# Initial parameters and time vector
fwhm = 0.2  # seconds
wave_time = np.arange(0, 2 * srate - 1) / srate
wave_time = wave_time - np.mean(wave_time)
half_wave = len(wave_time) // 2 + 1

# Compute convolution N's
n_conv = len(timevec) + len(wave_time) - 1

# Create wavelet and compute its spectrum
cmw = np.exp(1j * 2 * np.pi * 42 * wave_time) * np.exp(
    -4 * np.log(2) * wave_time**2 / fwhm**2
)
cmwX = np.fft.fft(cmw, n_conv)
cmwX = cmwX / np.max(cmwX)

# Run convolution
comp_sig = np.fft.ifft(np.fft.fft(erp, n_conv) * cmwX)
comp_sig = comp_sig[half_wave - 1 : -half_wave + 1]

cmw_amp = 2 * np.abs(comp_sig)

# %% Create an FIR filter at 42 Hz
# Create filter parameters
# One-sided Hz
filter_width = 7
# filter_width = 3  # Better results
center_freq = 42

# fir1 parameters
f_bounds = [center_freq - filter_width, center_freq + filter_width]
f_bounds = np.array(f_bounds) / (srate / 2)
order = 300

# Create the filter kernel
filt_kern = firwin(
    order + 1,
    f_bounds,
    pass_zero=False,
)

# Apply the filter
filt_sig = filtfilt(filt_kern, 1, erp)

# Extract amplitude time series using hilbert
fh_amp = np.abs(hilbert(filt_sig))

# %% Plotting
plt.figure()
plt.plot(timevec, cmw_amp, label='Wavelet convolution')
plt.plot(timevec, fh_amp, label='Filter-Hilbert')
plt.xlim([timevec[0], timevec[-1]])
plt.legend()
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (μV)')
plt.show()

# TO DO: Based on visual inspection, modify the FIR parameters to make
#        the two results as close as possible.

# %% 6) Wavelet convolution for all channels and visualize with tfviewerx

# So far, we've been doing time-frequency analysis one channel at a time.
# Now we will do it for all channels, and visualize the results using
# tfviewerx. Make sure to temporally downsample the results after convolution!

# Load the .mat file
mat = loadmat(os.path.join('..', 'data', 'sampleEEGdata.mat'))
# Get the time points
times = mat['EEG'][0][0][14][0]
# Get the epoched data of shape (channels x samples x trials)
data = mat['EEG'][0][0][15]
# Get the sampling frequency
srate = mat['EEG'][0][0][11][0][0]

# Get the number of samples
n_chs, n_samples, n_trials = data.shape

# Downsampled time points
times_to_save = np.arange(-250, 1251, 25)
t_idx = np.array([np.argmin(np.abs(times - time)) for time in times_to_save])

# Baseline time boundaries
base_times = np.array([-500, -200])
base_idx = np.array([np.argmin(np.abs(times - time)) for time in base_times])

# Time-frequency analysis parameters
freq_range = [2, 40]  # Extract only these frequencies (in Hz)
n_freqs = 33  # Number of frequencies between lowest and highest

# Set up convolution parameters
freqs = np.linspace(freq_range[0], freq_range[1], n_freqs)
wave_time = np.arange(-2, 2, 1 / srate)
n_kern = len(wave_time)
n_conv = n_samples * n_trials + n_kern - 1
half_wave = (len(wave_time) - 1) // 2
fwhms = np.linspace(0.5, 0.2, n_freqs)

# Initialize TF matrix
tf = np.zeros((n_chs, n_freqs, len(t_idx)))
tf_raw = np.zeros((n_chs, n_freqs))

# Compute the spectrum of each channel
dataX = np.fft.fft(data.reshape(n_chs, -1, order='F'), n_conv, axis=1)

# Create wavelets and perform TF decomposition
for fi in range(n_freqs):
    # Time-domain wavelet
    cmw = np.exp(2 * 1j * np.pi * freqs[fi] * wave_time) * np.exp(
        (-4 * np.log(2) * wave_time**2) / fwhms[fi] ** 2
    )
    cmwX = np.fft.fft(cmw, n_conv)
    cmwX /= np.max(cmwX)

    # Convolution across channels
    for chani in range(n_chs):
        comp_sig = np.fft.ifft(dataX[chani, :] * cmwX)
        comp_sig = comp_sig[half_wave + 1 : -half_wave].reshape(
            n_samples, n_trials, order='F'
        )
        comp_sig *= 2 / n_samples

        # Power time series
        pwr = np.mean(np.abs(comp_sig) ** 2, axis=1)

        # Raw power
        tf_raw[chani, fi] = np.mean(pwr)

        # Baseline-normalized power
        tf[chani, fi, :] = 10 * np.log10(
            pwr[t_idx] / np.mean(pwr[base_idx[0] : base_idx[1]])
        )


# Q: Is the double-loop really the smartest way to set this up? Why?
# A: For a small number of channels or a small dataset the second loop is not necessary.
# Q: What if you had multiple conditions? How would you modify the code?
# A: pwr = np.mean(np.abs(comp_sig) ** 2, axis=1) would change to something like
#    pwr1 = np.mean(np.abs(comp_sig[:, cond==1]) ** 2, axis=1) and then store the
#    results like tf_raw[chani, fi, 1] = np.mean(pwr1).

# %% 7) Compare wavelet convolution and mean over time with static FFT

# Adjust the code from the previous exercise to save the static spectrum
# without baseline normalization.
# Then implement a static FFT (like what you learned in the previous
# section of the course) to get power from one electrode.
# Compare the static power spectrum and the time-averaged TF power spectrum
# on the same graph.

# Focus on one channel
ch = 30

# Get power from the FFT
fft_pwr = (2 * np.abs(np.fft.fft(data[ch, :, :], axis=0)) / n_samples) ** 2
fft_pwr = np.mean(fft_pwr, 1)

# Define a vector of frequencies
hz = np.linspace(0, srate / 2, n_samples // 2 + 1)

# Plot!
plt.figure()
plt.plot(
    freqs, 2 * n_samples * tf_raw[ch, :] * 100, label='Average Time-Frequency Power'
)
plt.plot(hz, fft_pwr[: len(hz)], label='Static FFT Power')

# Make the plot look nicer
plt.xlim(-0.001, freqs[-1])
plt.ylim(0, 6)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power ($\mu V^2$)')
plt.title(f'Data from Channel {ch+1}')
plt.legend()
plt.show()

# Q: Are the two lines on the same scale? (No.)
# A: No, the average time-frequency power has to be scaled up.
# Q: Look through the code from the previous section to figure out
#    where these differences come from. How many scaling factors can you fix?
#    (Reminder that scaling factors are often weird and arbitrary
#    the shape of the spectrum is more important.)
# A: Yeah, maybe another time.
# Q: What do these results tell you about static vs. dynamic
#             spectral analyses?
# A: Dynamic spectral analyses are smoother in general.

# %%
