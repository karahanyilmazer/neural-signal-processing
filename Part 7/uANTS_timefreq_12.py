# %%
#   COURSE: Neural signal processing and analysis: Zero to hero
#  SECTION: Time-frequency analysis
#  TEACHER: Mike X Cohen, sincxpress.com
#    VIDEO: Scale-free dynamics via detrended fluctuation analysis

# !%matplotlib qt
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft, ifft
from scipy.io import loadmat
from scipy.signal import detrend, hilbert

# Set figure settings
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import set_fig_dpi, set_style

set_fig_dpi(), set_style()


# %%
def filterFGx(data, srate, f, fwhm, show_plot=False):
    """
    Narrow-band filter via frequency-domain Gaussian

    Parameters
    ----------
    data : array
        Input data, 1D or 2D (channels x time).
    srate : float
        Sampling rate in Hz.
    f : float
        Peak frequency of filter.
    fwhm : float
        Standard deviation of filter, defined as full-width at half-maximum in Hz.
    show_plot : bool, optional
        Set to True to show the frequency-domain filter shape.

    Returns
    -------
    filt_data : array
        Filtered data.
    emp_vals : list
        The empirical frequency and FWHM (in Hz and in ms).
    """

    # Input check
    if data.shape[0] > data.shape[1]:
        # raise ValueError(
        #     'Data dimensions may be incorrect. Data should be channels x time.'
        # )
        pass

    if (f - fwhm) < 0:
        # raise ValueError('Increase frequency or decrease FWHM.')
        pass

    if fwhm <= 0:
        raise ValueError('FWHM must be greater than 0.')

    # Frequencies
    hz = np.linspace(0, srate, data.shape[1])

    # Create Gaussian
    s = fwhm * (2 * np.pi - 1) / (4 * np.pi)  # Normalized width
    x = hz - f  # Shifted frequencies
    fx = np.exp(-0.5 * (x / s) ** 2)  # Gaussian
    fx = fx / np.abs(np.max(fx))  # Gain-normalized

    # Filter
    filt_data = 2 * np.real(ifft(fft(data, axis=1) * fx, axis=1))

    # Compute empirical frequency and standard deviation
    idx = np.argmin(np.abs(hz - f))
    emp_vals = [
        hz[idx],
        hz[idx - 1 + np.argmin(np.abs(fx[idx:] - 0.5))]
        - hz[np.argmin(np.abs(fx[:idx] - 0.5))],
    ]

    # Also temporal FWHM
    tmp = np.abs(hilbert(np.real(np.fft.fftshift(ifft(fx)))))
    tmp = tmp / np.max(tmp)
    tx = np.arange(data.shape[1]) / srate
    idxt = np.argmax(tmp)
    emp_vals.append(
        (
            tx[idxt - 1 + np.argmin(np.abs(tmp[idxt:] - 0.5))]
            - tx[np.argmin(np.abs(tmp[:idxt] - 0.5))]
        )
        * 1000
    )

    # Inspect the Gaussian (turned off by default)
    if show_plot:
        plt.figure()
        plt.subplot(211)
        plt.plot(hz, fx, 'o-')
        plt.plot(
            [
                hz[np.argmin(np.abs(fx[:idx] - 0.5))],
                hz[idx - 1 + np.argmin(np.abs(fx[idx:] - 0.5))],
            ],
            [
                fx[np.argmin(np.abs(fx[:idx] - 0.5))],
                fx[idx - 1 + np.argmin(np.abs(fx[idx:] - 0.5))],
            ],
            'k--',
        )
        plt.xlim([max(f - 10, 0), f + 10])
        plt.title(
            f'Requested: {f:.2f}, {fwhm:.2f} Hz; Empirical: {emp_vals[0]:.2f}, {emp_vals[1]:.2f} Hz'
        )
        plt.xlabel('Frequency (Hz)'), plt.ylabel('Amplitude Gain')

        plt.subplot(212)
        tmp1 = np.real(np.fft.fftshift(ifft(fx)))
        tmp1 = tmp1 / np.max(tmp1)
        tmp2 = np.abs(hilbert(tmp1))
        plt.plot(tx, tmp1, tx, tmp2)
        plt.xlabel('Time (s)'), plt.ylabel('Amplitude Gain')

        plt.tight_layout()
        plt.show()

    return filt_data, emp_vals, fx


# %%
mat = loadmat(os.path.join('..', 'data', 'dfa_data.mat'))
srate = mat['srate'][0, 0]
times = mat['timevec'][0]
x = mat['x'][0].reshape(1, -1)

# Try with narrowband amplitude time series
x_filt = filterFGx(x, srate, f=10, fwhm=5, show_plot=False)[0][0]
x_filt = np.abs(hilbert(x_filt))

# %%
# Create data with DFA=0.5
N = len(times)
rand_noise = np.random.randn(N)

# Setup parameters
n_scales = 20
ranges = np.round(N * np.array([0.01, 0.2])).astype(int)
scales = np.ceil(
    np.logspace(np.log10(ranges[0]), np.log10(ranges[1]), n_scales)
).astype(int)
rms_vals = np.zeros((2, n_scales))


# Plot the two signals
plt.figure()
plt.subplot(221)
plt.plot(times, rand_noise)
plt.title('Signal 1: White Noise')
plt.ylabel('Amplitude (a.u.)')

plt.subplot(222)
plt.plot(times, x_filt)
plt.title('Signal 2: Real Data')

# Integrate and mean-center the signals
rand_noise = np.cumsum(rand_noise - np.mean(rand_noise))
x_dfa = np.cumsum(x_filt - np.mean(x_filt))

# Show those time series for comparison
plt.subplot(223)
plt.plot(times, rand_noise)
plt.title('Integrated Noise')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (a.u.)')

plt.subplot(224)
plt.plot(times, x_dfa)
plt.title('Integrated Signal')
plt.xlabel('Time (s)')

plt.tight_layout()
plt.show()

# %%
# Compute RMS over different time scales
for scale_i in range(n_scales):
    # Number of epochs for this scale
    n = N // scales[scale_i]

    # Compute RMS for the random noise
    epochs = rand_noise[: n * scales[scale_i]].reshape(n, scales[scale_i]).T
    detrended_epochs = detrend(epochs, axis=0)
    rms_vals[0, scale_i] = np.mean(np.sqrt(np.mean(detrended_epochs**2, axis=1)))

    # Repeat for the signal
    epochs = x_dfa[: n * scales[scale_i]].reshape(n, scales[scale_i]).T
    detrended_epochs = detrend(epochs, axis=0)
    rms_vals[1, scale_i] = np.mean(np.sqrt(np.mean(detrended_epochs**2, axis=1)))

# %%
# Fit a linear model to quantify scaling exponent
A = np.vstack([np.ones(n_scales), np.log10(scales)]).T
dfa1 = np.linalg.lstsq(A, np.log10(rms_vals[0, :]), rcond=None)[0]
dfa2 = np.linalg.lstsq(A, np.log10(rms_vals[1, :]), rcond=None)[0]
# dfa1 = np.linalg.solve(A.T @ A, A.T @ np.log10(rms_vals[0, :]).T)  # Same as above
# dfa2 = np.linalg.solve(A.T @ A, A.T @ np.log10(rms_vals[1, :]).T)  # Same as above

# Plot the 'linear' fit (in log-log space)
plt.figure()

# Plot results for white noise
plt.plot(
    np.log10(scales),
    np.log10(rms_vals[0, :]),
    'rs',
    linewidth=2,
    markerfacecolor='w',
    markersize=10,
    label='Data (Noise)',
)
plt.plot(
    np.log10(scales),
    dfa1[0] + dfa1[1] * np.log10(scales),
    'r--',
    label=f'Fit (DFA={dfa1[1]:.3f})',
)

# Plot results for the real signal
plt.plot(
    np.log10(scales),
    np.log10(rms_vals[1, :]),
    'bs',
    linewidth=2,
    markerfacecolor='w',
    markersize=10,
    label='Data (Signal)',
)
plt.plot(
    np.log10(scales),
    dfa2[0] + dfa2[1] * np.log10(scales),
    'b--',
    label=f'Fit (DFA={dfa2[1]:.3f})',
)

plt.title('Comparison of Hurst Exponent for Different Noises')
plt.xlabel('Data Scale (log)')
plt.ylabel('RMS (log)')
plt.legend()

plt.show()

# %%
# DFA scanning through frequencies

# Parameters and initializations
freqs = np.linspace(1, 40, 80)
dfas = np.zeros(len(freqs))
rms = np.zeros(n_scales)

for fi in range(len(freqs)):
    # Get power time series
    x_filt = filterFGx(x, srate, f=freqs[fi], fwhm=5, show_plot=False)[0][0]
    x_filt = np.abs(hilbert(x_filt))

    # Integrate the mean-centered signal
    x_dfa = np.cumsum(x_filt - np.mean(x_filt))

    # Compute RMS over different time scales
    for scale_i in range(n_scales):
        # Number of epochs for this scale
        n = N // scales[scale_i]

        # Compute RMS for this scale
        epochs = x_dfa[: n * scales[scale_i]].reshape(n, scales[scale_i]).T
        detrended_epochs = detrend(epochs, axis=0)
        rms[scale_i] = np.mean(np.sqrt(np.mean(detrended_epochs**2, axis=1)))

    dfa = np.linalg.lstsq(A, np.log10(rms), rcond=None)[0]
    # dfa = np.linalg.solve(A.T @ A, A.T @ np.log10(rms).T)
    dfas[fi] = dfa[1]

plt.figure()
plt.plot(freqs, dfas, 'ko-', markerfacecolor='w')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Hurst Exponent')
plt.show()

# %%
