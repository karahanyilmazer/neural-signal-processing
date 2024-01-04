# %%
#   COURSE: Neural signal processing and analysis: Zero to hero
#  SESSION: Static spectral analyses
#  TEACHER: Mike X Cohen, sincxpress.com
#    VIDEO: Complex numbers and Euler's formula

# !%matplotlib inline
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

# Set figure settings
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import set_fig_dpi, set_style

set_fig_dpi(), set_style(notebook=True, grid=True)
# %%
# There are several ways to create a complex number
z = 4 + 3j
z = 4 + 3 * 1j
z = complex(4, 3)

print(f'Real part is {z.real} and imaginary part is {z.imag}.')

# Beware of this common programming error:
i = 2
zz = 4 + 3 * i

# Plot the complex number
fig, ax = plt.subplots()
plt.scatter(z.real, z.imag)

ax.set_aspect('equal', adjustable='box')  # Make the axes square
plt.xlim([-5, 5])
plt.ylim([-5, 5])
plt.vlines(0, -5, 5, 'black')
plt.hlines(0, -5, 5, 'black')
plt.xticks(np.arange(-5, 6))
plt.yticks(np.arange(-5, 6))
plt.xlabel('Real Axis')
plt.ylabel('Imaginary Axis')
plt.title(f'{int(z.real)} + {int(z.imag)}i on the Complex Plane')
plt.show()

# %% Euler's formula and the complex plane

# Use Euler's formula to plot vectors
m = 4
k = np.pi / 3
comp_num = m * np.exp(1j * k)

# Extract magnitude and angle
mag = np.abs(comp_num)
phs = np.angle(comp_num)

# Create a square figure
fig, ax = plt.subplots()
ax.set_aspect('equal', adjustable='box')

# Plot the unit circle
x = np.linspace(-np.pi, np.pi, 100)
plt.plot(np.cos(x), np.sin(x), 'gray')
# Plot the complex number
plt.plot([0, comp_num.real], [0, comp_num.imag], '-o')

plt.xlim([-5, 5])
plt.ylim([-5, 5])
plt.vlines(0, -5, 5, 'black')
plt.hlines(0, -5, 5, 'black')
plt.xticks(np.arange(-5, 6))
plt.yticks(np.arange(-5, 6))
plt.xlabel('Real Axis')
plt.ylabel('Imaginary Axis')
rect_str = f'{np.round(comp_num.real, 2)} + {np.round(comp_num.imag, 2)}i'
polar_str = f'{np.round(mag, 2)}e^{{{np.round(phs, 2)}i}}'
plt.title(f'Rectangular: {rect_str}\nPolar: ${polar_str}$')
plt.show()
