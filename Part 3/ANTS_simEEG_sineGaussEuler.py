# %%
#     COURSE: Solved challenges in neural time series analysis
#    SECTION: Simulating EEG data
#      VIDEO: The three important equations (sine, Gaussian, Euler's)
# Instructor: sincxpress.com

import numpy as np
import matplotlib.pyplot as plt

# %% SINE WAVES

# Define some variables
freq  = 2                         # Frequency in Hz
srate = 1000                      # Sampling rate in Hz
time  = np.arange(-1, 1, 1/srate) # Time vector in seconds
amp   = 2
phase = np.pi/3

# Create the sine wave
sine_wave = amp * np.sin(2 * np.pi * time * freq + phase)

# Plot the sine wave
plt.figure()
plt.plot(time, sine_wave)
plt.xlabel('Time (in s)')
plt.ylabel('Amplitude (arb. units)')
plt.show()

# %% SUM OF SINE WAVES
# Create a few sine waves and sum them up

# Sampling rate in Hz
srate = 1000

# List some frequencies
freqs = [3, 10, 5, 15, 35]

# List some random amplitudes... make sure there are the same number of
# amplitudes as there are frequencies!
amps = [5, 15, 10, 5, 7]

# Phases... list some random numbers between -pi and pi
phases = [np.pi/7, np.pi/8, np.pi, np.pi/2, -np.pi/4]

# Define time
np.arange(-1, 1, 1/srate)

# Loop through frequencies and create sine waves
sine_waves = np.zeros((len(freqs), len(time)))
for i, (amp, freq, phase) in enumerate(zip(amps, freqs, phases)):
    sine_waves[i, :] = amp * np.sin(2*np.pi*time*freq + phase)

# Plot the result
plt.figure()
plt.plot(time,sum(sine_waves))
plt.title('Sum of Sine Waves')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (arb. units)')
plt.xlim(min(time)*1.05, max(time)*1.05)

# Plot each wave separately
_, axs = plt.subplots(len(freqs), 1)
for ax, wave in zip(axs, sine_waves):
    ax.plot(time, wave)
    ax.set_xlim(min(time)*1.05, max(time)*1.05)
    ax.set_ylim(-max(amps)*1.05, max(amps)*1.05)
    if ax != axs[-1]:
        ax.set_xticks([])
axs[0].set_title('Individual Sine Components')
axs[-1].set_xlabel('Time (s)')

# %% GAUSSIAN

# Simulation parameters
srate     = 1000
time      = np.arange(-2, 2+1/srate, 1/srate)
peak_time = 1 
amp       = 45
fwhm      = 0.9

# Create the Gaussian
gauss = amp * np.exp(-(4*np.log(2)*(time-peak_time)**2) / fwhm**2)

# Empirical FWHM
gauss_norm = gauss/max(gauss)
mid_idx    = np.argmin(np.abs(time - peak_time))
post5      = mid_idx + np.argmin(np.abs(gauss_norm[mid_idx:] - 0.5))
pre5       = np.argmin(np.abs(gauss_norm[:mid_idx] - 0.5))
emp_fwhm   = time[post5] - time[pre5]

plt.figure()
plt.plot(time, gauss)
plt.plot([time[pre5], time[post5]], [gauss[pre5], gauss[post5]], marker='o', ls='-')
plt.plot([time[pre5], time[pre5]], [0, gauss[pre5]], 'orange', ls='--')
plt.plot([time[post5], time[post5]], [0, gauss[post5]], 'orange', ls='--')
plt.title(f'Requested FWHM: {fwhm} s, Empirical FWHM: {emp_fwhm} s')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.ylim(0, amp*1.05)
plt.show()

# %% EULER'S FORMULA
r = 2.4
theta = 3*np.pi/4

comp_val = r * np.exp(1j * theta)

fig = plt.figure()
ax1 = plt.subplot(121, projection='polar')
ax2 = plt.subplot(122)

ax1.plot([0, theta], [0, r])
ax1.plot(theta, r)
ax1.set_title('Polar Plane')

ax2.scatter(comp_val.real, comp_val.imag)
ax2.set_xlim(-abs(comp_val), abs(comp_val))
ax2.set_ylim(-abs(comp_val), abs(comp_val))
ax2.set_aspect('equal', adjustable='box')
ax2.set_title('Cartesian Plane')
ax2.set_xlabel('Real')
ax2.set_ylabel('Imag')
ax2.grid()

plt.tight_layout()
plt.show()
