import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import Circle
from math import ceil
from matplotlib import ticker

import os

class PupilPlot:
    """for plotting unstructured intensity data from rays"""

    def __init__(self, x, y, Ix, Iy):
        self.x = x
        self.y = y
        self.data_x = Ix
        self.data_y = Iy
        self.data_total = Ix + Iy
        self.figure = plt.figure()

    def get_default_plot_options(self):
        raise NotImplementedError()

    def plot(
            self, title,
            scale_range=None,
            rotate_90=False,
            caption_text="",
            pupil_boundary_radius=None):
        """tricontourf for plotting, and a polar transformation"""

        if len(self.x) < 4:
            print("Not enough points to plot pupil, skipping")
            return

        (x, y) = self.x, self.y
        (data_x, data_y, data_total) = self.data_x, self.data_y, self.data_total

        # a lot of this is attempts at automating tick range calculations, i gave up
        max_range = np.max(np.concatenate((data_x, data_y, data_total)))
        max_range = np.round(max_range * 20) / 20  # 0.05 intervals
        if scale_range is None:
            min_range = np.min(np.concatenate((data_x, data_y, data_total)))
            min_range = np.round(min_range * 20) / 20

        max_r = np.max([np.max(x), np.max(y), abs(np.min(x)), abs(np.min(y))])
        print("pupil_boundary_radius", pupil_boundary_radius, "max_r", max_r)

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
        self.add_heatmap_plot(fig, ax, x, y, data_x, pupil_radius=max_r,
                              min_range=None, max_range=None, title="X intensity")

        ax = fig.add_subplot(1, n_plots, 2)
        self.add_heatmap_plot(fig, ax, x, y, data_y, pupil_radius=max_r,
                              min_range=None, max_range=None, title="Y intensity")

        ax = fig.add_subplot(1, n_plots, 3)
        self.add_heatmap_plot(fig, ax, x, y, data_total, pupil_radius=max_r,
                              min_range=None, max_range=None, title="Total intensity")

        fig.suptitle(title, wrap=True)
        fig.set_layout_engine('tight')
        plt.show()

        self.figure = fig
        return fig

    def save(self, save_dir):
        save_dir = os.path.normpath(save_dir)
        dirname = os.path.dirname(save_dir)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        print("Saving in %s" % save_dir)
        self.figure.savefig(save_dir, bbox_inches="tight")

    def add_heatmap_plot(self, fig, ax, x, y, data, pupil_radius=None,
                         min_range=None, max_range=None, n_ticks=5, title=""):

        cmap = matplotlib.colormaps['autumn_r']
        cbar_ticks = None
        levels = None

        if min_range is None or max_range is None:  # autoscale
            levels = 257
            cbar_ticks = ticker.MaxNLocator(nbins=5, prune="both")
        else:
            levels = np.linspace(min_range, max_range, 257)
            if min_range > max_range:
                min_range = max_range

            range_diff = max_range - min_range
            power = ceil(np.log10(range_diff))
            n_ticks = np.round(range_diff * 10 ** (-power))
            cbar_ticks = np.linspace(min_range, max_range, n_ticks)

        if pupil_radius is None:
            pupil_radius = np.max([np.max(x), np.max(y), abs(np.min(x)), abs(np.min(y))])

        # r and phi for circle that shows NA/edge of pupil
        r_line = np.array([pupil_radius] * 100)
        phi_line = np.linspace(0, 2 * np.pi, 100)

        # auto-triangulating contour plot, takes unstructured data
        pc1 = ax.tricontourf(x, y, data, title="", cmap=cmap, levels=levels,
                             vmin=min_range, vmax=max_range, extend='both')

        plt.plot(r_line * np.cos(phi_line), r_line * np.sin(phi_line), color="k", zorder=2)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(title)
        fig.colorbar(pc1, ax=ax, fraction=0.04, pad=0.15, ticks=cbar_ticks, location="bottom")
