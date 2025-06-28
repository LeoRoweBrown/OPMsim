import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from math import ceil
from matplotlib import ticker

# Module for plotting intensity of exit pupil. General purpose

def plot_pupil_intensity(
        x, y,
        data_x, data_y,
        title="Exit pupil intensity",
        scale_range=[None, None],
        rotate_90=False,
        caption_text="",
        pupil_boundary_radius=None):
    """tricontourf for plotting, and a polar transformation"""

    if len(x) < 4:
        print("Not enough points to plot pupil, skipping")
        return

    data_total = data_x + data_y

    # a lot of this is attempts at automating tick range calculations, i gave up
    min_range, max_range = scale_range

    if min_range is None:
        min_range = np.min(np.concatenate((data_x, data_y, data_total)))
        min_range = np.round(min_range * 20) / 20
    if max_range is None:
        max_range = np.max(np.concatenate((data_x, data_y, data_total)))
        max_range = np.round(max_range * 20) / 20  # 0.05 intervals

    max_r = np.max([np.max(x), np.max(y), abs(np.min(x)), abs(np.min(y))])

    if pupil_boundary_radius is not None:
        max_r = max([pupil_boundary_radius, max_r])

    # suppress strange floating points
    data_x[data_x < 1e-7] = 0
    data_y[data_y < 1e-7] = 0
    data_total[data_total < 1e-7] = 0

    # make new triangulation and contour map
    figsize = (7, 3)
    n_plots = 3

    pad = 0.04  # TODO add this padding to whatever needs it
    fig = plt.figure(figsize=figsize)
    fig.text(0.5, -0.05, caption_text, ha="center", fontsize=13)

    ax = fig.add_subplot(1, n_plots, 1)
    _add_heatmap_plot(fig, ax, x, y, data_x, pupil_radius=max_r,
                      min_range=min_range, max_range=max_range, title="X intensity")

    ax = fig.add_subplot(1, n_plots, 2)
    _add_heatmap_plot(fig, ax, x, y, data_y, pupil_radius=max_r,
                      min_range=min_range, max_range=max_range, title="Y intensity")

    ax = fig.add_subplot(1, n_plots, 3)
    _add_heatmap_plot(fig, ax, x, y, data_total, pupil_radius=max_r,
                      min_range=min_range, max_range=max_range, title="Total intensity")

    fig.suptitle(title, wrap=True, fontweight='bold')
    fig.set_layout_engine('tight')
    plt.show()

    return fig

def _add_heatmap_plot(fig, ax, x, y, data, pupil_radius=None,
                      min_range=0, max_range=0, n_ticks=5, title=""):

    cmap = matplotlib.colormaps['autumn_r']
    cbar_ticks = None
    levels = None

    # If min_range == max_range fall back to autoscale. Will raise an exception otherwise
    # default values of params invoke this condition.
    if abs(min_range - max_range) < 1e-6:  # autoscale
        levels = 257
        cbar_ticks = ticker.MaxNLocator(nbins=5, prune="both")
    else:
        levels = np.linspace(min_range, max_range, 257)
        if min_range > max_range:
            min_range = max_range

        cbar_ticks = np.linspace(min_range, max_range, n_ticks)
        cbar_ticks = np.round(cbar_ticks, 2)

    if pupil_radius is None:
        pupil_radius = np.max([np.max(x), np.max(y), abs(np.min(x)), abs(np.min(y))])

    # r and phi for circle that shows NA/edge of pupil
    r_line = np.array([pupil_radius] * 100)
    phi_line = np.linspace(0, 2 * np.pi, 100)

    # auto-triangulating contour plot, takes unstructured data
    pc1 = ax.tricontourf(x, y, data, title="", cmap=cmap, levels=levels,
                         vmin=min_range, vmax=max_range, extend='both')

    plt.plot(r_line * np.cos(phi_line), r_line * np.sin(phi_line), color="k", zorder=2, clip_on=False)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title)
    fig.colorbar(pc1, ax=ax, fraction=0.04, pad=0.10, ticks=cbar_ticks, location="bottom")
