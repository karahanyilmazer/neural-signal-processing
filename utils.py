import matplotlib.pyplot as plt
import scienceplots
from matplotlib.colors import LinearSegmentedColormap
from yaml import safe_load


def set_style(notebook=True, grid=False):
    # Set the style to science
    args = ['science', 'no-latex']
    if grid:
        args.append('grid')
    if notebook:
        args.append('notebook')
    plt.style.use(args)


def set_fig_dpi():
    # Set the figure dpi to 260
    plt.matplotlib.rcParams['figure.dpi'] = 260


def get_cmap(name):
    # Load the (parula) cmap
    with open(f'../{name}.yaml', 'r') as file:
        cmap = safe_load(file)['cmap']
        cmap = LinearSegmentedColormap.from_list(name, cmap)
    return cmap
