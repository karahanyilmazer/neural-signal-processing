import matplotlib.pyplot as plt
import scienceplots
from matplotlib.colors import LinearSegmentedColormap
from yaml import safe_load


def set_style():
    # Set the style to science
    plt.style.use(['science', 'notebook', 'no-latex'])


def set_fig_dpi():
    # Set the figure dpi to 260
    plt.matplotlib.rcParams['figure.dpi'] = 260


def get_cmap(name):
    # Load the (parula) cmap
    with open(f'../{name}.yaml', 'r') as file:
        cmap = safe_load(file)['cmap']
        cmap = LinearSegmentedColormap.from_list(name, cmap)
    return cmap
