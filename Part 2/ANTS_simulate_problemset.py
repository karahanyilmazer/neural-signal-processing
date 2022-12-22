# %%
import numpy as np
import matplotlib.pyplot as plt


# %%
def plot_EEG(times, data, ch_idx=0):
    times = times.copy() * 1000
    mosaic = [['ERP', 'ERP'], ['PSD', 'TF']]
    fig = plt.figure(constrained_layout=True)
    ax_dict = fig.subplot_mosaic(mosaic)

    ax_dict['ERP'].plot(times, data[ch_idx, :, :], color='grey', alpha=0.1)
    ax_dict['ERP'].plot(times,
                        np.mean(data[ch_idx, :, :], axis=1),
                        color='black')
    ax_dict['ERP'].set_title(f'ERP From Channel {ch_idx+1}')
    ax_dict['ERP'].set_xlabel('Time (ms)')
    ax_dict['ERP'].set_ylabel('Activity')
    ax_dict['ERP'].set_xlim(times.min(), times.max())

    freqs = np.linspace(0, srate, n_samples)
    if data.ndim == 3:
        # Perform FFT along the columns (trials)
        pw = np.mean((2 * np.abs(np.fft.fft(data[ch_idx, :, :], axis=0) / n_samples))**2,
                     axis=1)
    else:
        pw = (2 * np.abs(np.fft.fft(data[ch_idx, :], axis=0) / n_samples))**2
    ax_dict['PSD'].plot(freqs, pw)
    ax_dict['PSD'].set_title(f'Static Power Spectrum')
    ax_dict['PSD'].set_xlabel('Frequency (Hz)')
    ax_dict['PSD'].set_ylabel('Power')
    ax_dict['PSD'].set_xlim(0, 40)

    # Frequencies in Hz (hard-coded to 2 to 30 in 40 steps)
    freqs = np.linspace(2, 30, 40)
    # Number of wavelet cycles (hard-coded to 3 to 10)
    waves = 2 * (np.linspace(3, 10, len(freqs)) / (2 * np.pi * freqs))**2

    # Setup wavelet and convolution parameters
    wave_t = np.arange(-2, 2+1/srate, 1 / srate)
    half_w = int(np.floor(len(wave_t) / 2) + 1)
    n_conv = n_samples * n_trials + len(wave_t) - 1

    # Initialize the time-frequency matrix
    tf_mat = np.zeros((len(freqs), n_samples))

    # Spectrum of data
    data_fft = np.fft.fft(data[ch_idx, :, :].reshape(1, -1, order='F'), n_conv)

    for i in range(len(freqs)):
        wave_x = np.fft.fft(
            np.exp(2 * 1j * np.pi * freqs[i] * wave_t) *
            np.exp(-(wave_t**2) / waves[i]), n_conv)
        wave_x = wave_x / np.max(wave_x)

        amp_spec = np.fft.ifft(wave_x * data_fft)
        amp_spec = amp_spec[0][half_w-1:-half_w+1].reshape((n_samples, n_trials), order='F')

        tf_mat[i, :] = np.mean(np.abs(amp_spec), axis=1)

    ax_dict['TF'].contourf(times, freqs, tf_mat, 40, cmap='jet')
    ax_dict['TF'].set_title(f'Time-Frequency Plot')
    ax_dict['TF'].set_xlabel('Time (ms)')
    ax_dict['TF'].set_ylabel('Frequency (Hz)')
    ax_dict['TF'].set_xlim(times.min(), times.max())

    plt.show()

# %% [1. WHITE NOISE]
# Define the sampling rate
srate = 500  # in Hz
duration = 3  # in s
n_samples = srate * duration
n_trials = 30
n_chans = 3
amp_factor = 1

# Create the time vector
times = np.arange(n_samples) / srate

# Create data as white noise
data = amp_factor * np.random.randn(n_chans, n_samples, n_trials)

plot_EEG(times, data, ch_idx=1)

# Q: What is the effect of noise amplitude on the resulting graphs?
# A: Amplitude changed by the factor, power spectrum changed by factor^2
# Q: Do the results change if you use normally distributed vs. uniformly distributed noise?
# A: Yes, there is a large DC offset which can be removed by subtracting 0.5 from the signal
# Q: Are the results different for different channels? Why or why not?
# A: Technically yes, but here not much


# %% [2. PINK NOISE]
# Exponential decay
exp_decay = 50

# Initalize an array for the data
data = np.zeros((n_chans, n_samples, n_trials))

# Iterate over the channels
for ch in range(n_chans):
    # Iterate over the trials
    for trial in range(n_trials):
        # Create a signal in the frequency domain and transform it into the time domain
        # Generate one-sided 1/f amplitude spectrum
        amp_spec = np.random.rand(1, n_samples) * np.exp(-(np.arange(n_samples))/exp_decay)
        
        # Get the shape of the amplitude spectrum
        m, n = amp_spec.shape

        # Fourier coefficients as amplitudes times random phases
        four_coef = amp_spec * np.exp(1j*2*np.pi*np.random.rand(m, n))
        
        # Inverse Fourier transform to create the noise
        data[ch, :, trial] = np.fft.ifft(four_coef).real

plot_EEG(times, data, ch_idx=1)

# Q: Which looks more like real EEG data: white or pink noise? Why do you think this is?
# A: Pink nois, as the power spectrum looks closer to EEG data
# Q: Which values of variable 'exp_decay' make the data look most like real EEG data?
# A: Around 50

# %% [3. ONGOING STATIONARY]
# The goal here is to create a dataset with ongoing sinewaves
# There should be multiple sine waves simultaneously in each channel/trial

# List of frequencies and corresponding amplitudes
freqs = [3, 5, 16]  # in Hz
amps = [3, 4, 5]    # in arbitrary units
phases = [0, 0, 0]

# Initalize an array for the data
data = np.zeros((n_chans, n_samples, n_trials))

# Loop over channels and trials
for ch in range(n_chans):
    for trial in range(n_trials):
        
        # Note that here the signal is created in the time domain, unlike in the previous example.
        # Some signals are easier to create in the time domain; others in the frequency domain.
        
        # Create a multicomponent sine wave
        sine_wave = np.zeros((1, n_samples))

        for freq, amp, phase in zip(freqs, amps, phases):
            sine_wave += amp * np.sin(2 * np.pi * freq * times + phase)
            # sine_wave += amp * np.sin(2 * np.pi * freq * times + phase + np.random.randn() * 2 * np.pi)
        
        # Data as a sine wave plus noise
        data[ch, :, trial] = sine_wave + np.random.randn(*sine_wave.shape)

plot_EEG(times, data, ch_idx=1)

# Q: What can you change in the code above to make the EEG activity non-phase-locked over trials?
# A: Randomize the phases
# Q: Which of the plots look different for phase-locked vs. non-phase-locked?
#    (Hint: plot them in different figures to facilitate comparison.)
#    Are you surprised about the differences?
# A: Time series and time-frequency plots look different. TF of single trial still has the erronous
#    "beads" but they get canceled out due to phase shifts in the non-phase-locked case.
# Q: Are all frequencies equally well represented in the 'static' and 'dynamic' power spectra?
#    Can you change the parameters to make the spectral peaks more or less visible in the two plots?
# A: All frequencies represented in the static power spectrum but not in the TF plot. Separating the
#    frequencies help with the TF plot.

# %% [4. ONGOING NON-STATIONARY]
# Here you want to create narrowband non-stationary data. 
# This is starting to be more "realistic" (in a signal-characteristic sense) for EEG data.

# Signal parameters in Hz
peak_freq_1 = 21
peak_freq_2 = 8
fwhm        = 5

# Frequencies
freqs = np.linspace(0, srate, n_samples)

# Create frequency-domain Gaussian
s  = fwhm*(2*np.pi-1)/(4*np.pi)   # normalized width
x  = freqs - peak_freq_1          # shifted frequencies
fg_1 = np.exp(-0.5*(x/s)**2)      # gaussian

x  = freqs - peak_freq_2          # shifted frequencies
fg_2 = np.exp(-0.5*(x/s)**2)      # gaussian

# Re-initialize EEG data
data = np.zeros((n_chans, n_samples, n_trials))

for ch in range(n_chans):
    for trial in range(n_trials):
        
        # As with previous simulations, don't worry if you don't understand the mechanisms;
        # that will be clear tomorrow. Instead, you can plot each step to try to build intuition.
        
        # Fourier coefficients of random spectrum
        fc = np.random.rand(1, n_samples) * np.exp(1j*2*np.pi*np.random.rand(1, n_samples))
        
        # Taper Fourier coefficients by the Gaussian
        fc = fc * fg_1 + fc * fg_2
        
        # Go back to time domain to get EEG data
        data[ch, :, trial] = np.fft.ifft(fc).real

plot_EEG(times, data, ch_idx=1)

# Q: What is the effect of FWHM on the results? Is larger or smaller more realistic?
# A: Small FWHM makes the power spectrum very narrow, large FWHM makes it too wide.
# Q: Can you modify the code to have narrowband activity at two different frequency ranges?
# A: Yes.
 
# %% [5. TRAINSIENTS: GAUSSIAN]

# All the exercises above were for ongoing signals. Now we move to transients.
# Start with a Gaussian.

# Gaussian parameters (in seconds)
peak_time = 1
width = 0.1

# Re-initialize EEG data
data = np.zeros((n_chans, n_samples, n_trials))

for ch in range(n_chans):
    for trial in range(n_trials):
        
        # Generate time-domain gaussian
        trial_peak = peak_time + np.random.randn() * 0.25
        gauss = np.exp(-(times-trial_peak)**2 / (2*width**2))
        
        # Data are the Gaussian
        data[ch , :, trial] = gauss

plot_EEG(times, data, ch_idx=1)

# Q: What happens if you add random jitter to the peaktime on each trial? 
# A: The peak spreads
# Q: How much jitter until the ERP is nearly gone?
# A: A factor of around 1 makes it very irregular

# %% [6. TRAINSIENTS: OSCILLATIONS WITH GAUSSIAN]

# Finally, we get to the most useful simulations for time-frequency analysis:
# time-limited narrow-band activity. This is done by multiplying a Gaussian with a sine wave.

# Sine wave frequency
sine_freq = 8

# Gaussian parameters (in seconds)
peak_time = 1
width = 0.2

# Re-initialize EEG data
data = np.zeros((n_chans, n_samples, n_trials))

for ch in range(n_chans):
    for trial in range(n_trials):
        
        # Generate time-domain gaussian
        trial_peak = peak_time + np.random.randn() / 5
        gauss = np.exp(-(times-trial_peak)**2 / (2*width**2))
        
        
        # Generate sine wave with same phase
        sine_wave = np.sin(2 * np.pi * sine_freq * times)
        # sine_wave = np.sin(2 * np.pi * sine_freq * times + np.random.randn() * 1 * np.pi)
        
    
        # Data are sine wave times Gaussian
        data[ch, :, trial] = gauss * sine_wave
        # data[ch, :, trial] = gauss * sine_wave + np.random.randn(*gauss.shape)

plot_EEG(times, data, ch_idx=1)

# Q: Do the results look realistic? What can you change to make it look even more EEG-like?
# A: Add noise to every possible step, lol
# Q: How can you modify the code to make the transient non-phase-locked?
#    Which of the three data plots are most affected by phase-locked vs. non-phase-locked?
# A: Add noise to the sine wave's phase. Only the ERP plot is affected.

# %% [7. GAUSSIAN + PINK NOISE]
# Exponential decay
exp_decay = 50

# Gaussian parameters (in seconds)
peak_time = 1
width = 0.1

# Initalize an array for the data
data = np.zeros((n_chans, n_samples, n_trials))

# Iterate over the channels
for ch in range(n_chans):
    # Iterate over the trials
    for trial in range(n_trials):

        # Generate time-domain gaussian
        trial_peak = peak_time + np.random.randn()/7.75
        gauss = np.exp(-(times-trial_peak)**2 / (2*width**2))

        # Create a signal in the frequency domain and transform it into the time domain
        # Generate one-sided 1/f amplitude spectrum
        amp_spec = np.random.rand(1, n_samples) * np.exp(-(np.arange(n_samples))/exp_decay)
        
        # Get the shape of the amplitude spectrum
        m, n = amp_spec.shape

        # Fourier coefficients as amplitudes times random phases
        four_coef = amp_spec * np.exp(1j*2*np.pi*np.random.rand(m, n))
        
        # Inverse Fourier transform to create the noise
        data[ch, :, trial] = 300 * np.fft.ifft(four_coef).real + gauss

plot_EEG(times, data, ch_idx=1)

# %% [8. TRANSIENT GAUSSIAN-WINDOWED NON-STATIONARY]

# Signal parameters in Hz
peak_freq = 21
fwhm      = 5

# Frequencies
freqs = np.linspace(0, srate, n_samples)

# Create frequency-domain Gaussian
s  = fwhm*(2*np.pi-1)/(4*np.pi) # normalized width
x  = freqs - peak_freq          # shifted frequencies
fg = np.exp(-0.5*(x/s)**2)      # gaussian

# Gaussian parameters (in seconds)
peak_time = 1
width = 0.2

# Re-initialize EEG data
data = np.zeros((n_chans, n_samples, n_trials))

for ch in range(n_chans):
    for trial in range(n_trials):
        
        # As with previous simulations, don't worry if you don't understand the mechanisms;
        # that will be clear tomorrow. Instead, you can plot each step to try to build intuition.
        
        # Fourier coefficients of random spectrum
        fc = np.random.rand(1, n_samples) * np.exp(1j*2*np.pi*np.random.rand(1, n_samples))
        
        # Taper Fourier coefficients by the Gaussian
        fc = fc * fg
        fc_real = np.fft.ifft(fc).real
        
        # Generate time-domain gaussian
        trial_peak = peak_time + np.random.randn() / 5
        gauss = np.exp(-(times-trial_peak)**2 / (2*width**2))

        # Go back to time domain to get EEG data
        data[ch, :, trial] = fc_real * gauss
        # data[ch, :, trial] = fc_real * gauss + np.random.randn(*gauss.shape)

plot_EEG(times, data, ch_idx=1)
