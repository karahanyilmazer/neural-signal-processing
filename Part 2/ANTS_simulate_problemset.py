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

        tmp = np.fft.ifft(wave_x * data_fft)
        tmp = tmp[0][half_w-1:-half_w+1].reshape((n_samples, n_trials), order='F')

        tf_mat[i, :] = np.mean(np.abs(tmp), axis=1)

    ax_dict['TF'].contourf(times, freqs, tf_mat, 40, cmap='jet')
    ax_dict['TF'].set_title(f'Time-Frequency Plot')
    ax_dict['TF'].set_xlabel('Time (ms)')
    ax_dict['TF'].set_ylabel('Frequency (Hz)')
    ax_dict['TF'].set_xlim(times.min(), times.max())

    plt.show()

# %%
# Define the sampling rate
srate = 500  # in Hz
duration = 3  # in s
n_samples = srate * duration
n_trials = 30
n_chans = 23
amp_factor = 1

# Create the time vector
times = np.arange(n_samples) / srate

# Create data as white noise
data = amp_factor * np.random.randn(n_chans, n_samples, n_trials)

plot_EEG(times, data, ch_idx=1)

# %%
