import numpy as np
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import matplotlib.tri as tri 
from scipy.spatial import Delaunay
from scipy.spatial import ConvexHull
import matplotlib

import os


class PupilPlotObject():
    """for plotting unstructured data at the moment"""
    def __init__(self, x, y, Ix, Iy):
        self.x = x
        self.y = y
        self.data_x = Ix
        self.data_y = Iy
        self.data_total = Ix + Iy
        self.figure = None

    def get_default_plot_options(self):
        raise NotImplementedError()
        
    def plot(self, title,
            show_prints=False, plot_arrow=None,
            fill_zeroes=False, scale_range=None,
            rotate_90=False, caption_text=None, max_r_in=None):
        """tricontourf for plotting, and a polar transformation"""

        if len(self.x) < 4:
            print("Not enough points to plot pupil, skipping")
            return

        dipole_alpha = None
        dipole_phi = None

        (x, y) = self.x, self.y
        (data_x, data_y, data_total) =  self.data_x,\
            self.data_y, self.data_total

        # print("data in plot: x", x, "y", y, "data x", data_x)
        print("scale range in _plot_unstructured", scale_range)

        max_for_scale = np.max(np.concatenate((data_x, data_y, data_total)))
        if scale_range is None:
            min_for_scale = np.min(np.concatenate((data_x, data_y, data_total)))
        else:
            try:
                min_for_scale = scale_range[0]
                if scale_range[1] is not None:
                    max_for_scale = scale_range[1]
            except TypeError as e:
                raise Exception(
                    "scale_range must be in the form [float, float], " + 
                    "but is %s" % scale_range
                    ) from e

        if min_for_scale > max_for_scale:
            min_for_scale = max_for_scale

        print("min for scale", min_for_scale)
        print("max for scale", max_for_scale)
        
        print("Filling background of polar plot with zeroes!")
        # get initial triangulation:
        # chull = ConvexHull(np.array((x,y)).T)  # get convex hull of data points
        delaunay = Delaunay(np.array((x,y)).T)  # get triangulation for simplex

        max_r = np.max([np.max(x), np.max(y), abs(np.min(x)), abs(np.min(y))])
        print("max_r_in", max_r_in, "max_r", max_r)

        if max_r_in is not None:
            max_r = max([max_r_in, max_r])

        # get a better estimate of maximum radius - see a diagram for this (ask me?)
        sorted_phi = np.sort(np.arctan2(y,x))
        diff_phi = \
            [abs(sorted_phi[i+1]-sorted_phi[i]) for i in range(len(sorted_phi)-1)]
        half_max_phi = max(diff_phi)/2
        max_r_reduced = max_r*np.cos(half_max_phi)
        if fill_zeroes:
            ## now create radius of zeroes
            #TODO: think about ways to stop bg points appearing outside the edges of the
            # actual data - sampling too high for bg
            r = np.linspace(0, max_r_reduced-1e-9,25)
            phi = np.linspace(0, 2*np.pi,100)
            r_m, phi_m = np.meshgrid(r, phi)
            x_m = r_m*np.cos(phi_m)
            y_m = r_m*np.sin(phi_m)
            x_bg = x_m.flatten()  # points used for background in triangulation
            y_bg = y_m.flatten()
            xy_bg = np.column_stack((x_bg,y_bg))
            bg_vals = np.zeros_like(x_bg)

            # remove points in the region
            # find bg points inside the triangulation and remove them:
            points_outside_tri = delaunay.find_simplex(xy_bg) == -1
            print("points outside tri", np.where(points_outside_tri==True))
            x_bg = x_bg[points_outside_tri]
            y_bg = y_bg[points_outside_tri]
            bg_vals = np.zeros_like(x_bg)

            # now add these points onto data
            x = np.append(x, x_bg)
            y = np.append(y, y_bg)

            data_x = np.append(data_x, bg_vals)
            data_y = np.append(data_y, bg_vals)
            data_total = np.append(data_total, bg_vals)

        # draw line instead
        r_line = np.array([max_r]*100)
        phi_line = np.linspace(0, 2*np.pi,100)
            

        # make new triangulation and contour map

        # plot scatter
        plt.figure()
        plt.scatter(x,y)
        plt.gca().axis('equal')
        plt.show()
        
        fig = plt.figure(figsize=[10,3]) 

        cbar_ticks = np.linspace(min_for_scale, max_for_scale, 11)
        cmap=matplotlib.cm.get_cmap('autumn_r')

        ax = fig.add_subplot(131)
        # print("I-field data", data_x)
        len(x) + 1
        levels = np.linspace(min_for_scale, max_for_scale, 257)
        if rotate_90:
            pc1 = ax.tricontourf(list(y), list(x), data_x, cmap=cmap,\
                levels=levels, vmin=min_for_scale,vmax=max_for_scale, extend='both')
        else:
            pc1 = ax.tricontourf(list(x), list(y), data_x, cmap=cmap,\
                levels=levels, vmin=min_for_scale,vmax=max_for_scale, extend='both')
        plt.plot(r_line*np.cos(phi_line), r_line*np.sin(phi_line), color='k', zorder=2)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title("X component intensity")
        
        fig.colorbar(pc1, ax=ax, fraction=0.04, pad=0.15, ticks=cbar_ticks)

        ### y
        ax2 = fig.add_subplot(132)
        if rotate_90:
            pc2 = ax2.tricontourf(list(y), list(x), data_y, cmap=cmap,\
                levels=levels, vmin=min_for_scale,vmax=max_for_scale, extend='both')
        else:
            pc2 = ax2.tricontourf(list(x), list(y), data_y, cmap=cmap,\
                levels=levels, vmin=min_for_scale,vmax=max_for_scale, extend='both')
        plt.plot(r_line*np.cos(phi_line), r_line*np.sin(phi_line), color='k',zorder=4)
        ax2.set_aspect('equal')
        ax2.axis('off')
        ax2.set_title("Y component intensity")

        fig.colorbar(pc2, ax=ax2, fraction=0.04, pad=0.15, ticks=cbar_ticks)

        ### total
        ax3 = fig.add_subplot(133)
        if rotate_90:
            pc3 = ax3.tricontourf(list(y), list(x), data_total, cmap=cmap,\
                levels=levels, vmin=min_for_scale,vmax=max_for_scale, extend='both')
        else:
            pc3 = ax3.tricontourf(list(x), list(y), data_total, cmap=cmap,\
                levels=levels, vmin=min_for_scale,vmax=max_for_scale, extend='both')
        plt.plot(r_line*np.cos(phi_line), r_line*np.sin(phi_line), color='k',zorder=5)
        ax3.set_aspect('equal')
        ax3.axis('off')
        ax3.set_title("Total intensity")

        fig.colorbar(pc3, ax=ax3, fraction=0.04, pad=0.15, ticks=cbar_ticks)
        # plt.subplots_adjust(wspace=0.45, hspace=None)
        fig.suptitle(title)

        fig.text(.5, -0.05, caption_text, ha='center', fontsize=14)

        fig.tight_layout()

        self.figure = fig
        return fig, (pc1, pc2, pc3)

    def save(self, save_dir):
        save_dir = os.path.normpath(save_dir)
        dirname =  os.path.dirname(save_dir)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        print("Saving in %s" % save_dir)
        self.figure.savefig(save_dir, bbox_inches='tight')