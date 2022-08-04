import numpy as np
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import matplotlib.tri as tri 
from scipy.spatial import Delaunay
from scipy.spatial import ConvexHull

import os


class PupilPlotObject():
    """for plotting unstructured data at the moment"""
    def __init__(self, x, y, Ix, Iy):
        self.x = x
        self.y = y
        self.data_x = Ix
        self.data_y = Iy
        self.data_total = Ix + Iy

    def plot(self, title="", show_prints=False, plot_arrow=None, projection="polar",
            unstructured_data=True, fill_zeroes=True):
        if len(self.x) < 4:
            print("Not enough points to plot pupil, skipping")
            return
        if unstructured_data:
            self._plot_unstructured(title, save_dir=None,
                file_name=None, show_prints=False, plot_arrow=None,
                projection="polar", fill_zeroes=fill_zeroes)
        else:
            raise NotImplementedError(
                "Only plotting for unstructured data using triangulation implemented")
        
    def _plot_unstructured(self, title, save_dir=None,
                file_name=None, show_prints=False, plot_arrow=None,
                projection="polar", fill_zeroes=True):
        """tricontourf for plotting, and a polar transformation"""

        dipole_alpha = None
        dipole_phi = None

        (x, y) = self.x, self.y
        (data_x, data_y, data_total) =  self.data_x,\
            self.data_y, self.data_total

        max_for_scale = np.max(np.concatenate((data_x, data_x, data_total)))
        min_for_scale = np.min(np.concatenate((data_x, data_x, data_total)))

        if fill_zeroes:
            # get initial triangulation:
            # chull = ConvexHull(np.array((x,y)).T)  # get convex hull of data points
            delaunay = Delaunay(np.array((x,y)).T)  # get triangulation for simplex

            max_r = np.max([np.max(x), np.max(y), abs(np.min(x)), abs(np.min(y))]) 
            print("max rad", max_r)

            ## now create radius of zeroes
            #TODO: think about ways to stop bg points appearing outside the edges of the
            # actual data - sampling too high for bg
            r = np.linspace(0, max_r,25)
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
            x_bg = x_bg[points_outside_tri]
            y_bg = y_bg[points_outside_tri]
            bg_vals = np.zeros_like(x_bg)

            print("length before:", np.shape(x), np.shape(y), np.shape(data_x))
            print("bg_vals", np.shape(bg_vals))
            print("x bg", np.shape(x_bg))
            # now add these points onto data
            x = np.append(x, x_bg)
            y = np.append(y, y_bg)
            data_x = np.append(data_x, bg_vals)
            data_y = np.append(data_y, bg_vals)
            data_total = np.append(data_total, bg_vals)

        # make new triangulation and contour map
        
        fig = plt.figure(figsize=[10,3]) 

        ax = fig.add_subplot(131)
        # print("I-field data", data_x)
        print(len(x), len(y), len(data_x))

        pc1 = ax.tricontourf(list(x), list(y), data_x,\
            levels=256, vmin=min_for_scale,vmax=max_for_scale)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title("X component intensity")
        
        fig.colorbar(pc1, ax=ax, fraction=0.04, pad=0.15)

        ### y
        ax2 = fig.add_subplot(132)
        pc2 = ax2.tricontourf(list(x), list(y), data_y,\
            levels=256, vmin=min_for_scale,vmax=max_for_scale)
        ax2.set_aspect('equal')
        ax2.axis('off')
        ax2.set_title("Y component intensity")

        fig.colorbar(pc2, ax=ax2, fraction=0.04, pad=0.15)

        ### total
        ax3 = fig.add_subplot(133)
        pc3 = ax3.tricontourf(list(x), list(y), data_total,\
         levels=256, vmin=min_for_scale,vmax=max_for_scale)
        ax3.set_aspect('equal')
        ax3.axis('off')
        ax3.set_title("Total intensity")

        fig.colorbar(pc3, ax=ax3, fraction=0.04, pad=0.15)
        # plt.subplots_adjust(wspace=0.45, hspace=None)
        fig.suptitle(title)

        fig.tight_layout()

        self.figure = fig
        return fig, (pc1, pc2, pc3)

    def save(self, save_dir):
        save_dir = os.path.normpath(save_dir)
        print("Saving in %s" % save_dir)
        self.figure.savefig(save_dir, bbox_inches='tight')