import numpy as np
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import matplotlib.tri as tri 
from scipy.spatial import Delaunay
from scipy.spatial import ConvexHull
import matplotlib
from matplotlib.patches import Circle
from math import ceil
import textwrap
from matplotlib import ticker

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
            rotate_90=False, caption_text=None, max_r_in=None,
            use_circle_path=False, add_autoscale_plots=False,
            font_sizes=[14,12,11], draw_NA_circle=None):
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

        # a lot of this is attempts at automating tick range calculations, i gave up
        n_ticks=6
        max_for_scale = np.max(np.concatenate((data_x, data_y, data_total)))
        max_for_scale = np.round(max_for_scale*20)/20
        if scale_range is None:
            min_for_scale = np.min(np.concatenate((data_x, data_y, data_total)))
            min_for_scale = np.round(min_for_scale*20)/20
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
            if len(scale_range) == 3:
                n_ticks=scale_range[2]

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
        print("max_r_reduced", max_r_reduced, "max_r", max_r)
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

        plt.figure()
        plt.hist(data_y)
        plt.show()

        # suppress strange floating points
        data_x[data_x < 1e-7] = 0
        data_y[data_y < 1e-7] = 0
        data_total[data_total < 1e-7] = 0

        # draw line instead
        r_line = np.array([max_r]*100)
        phi_line = np.linspace(0, 2*np.pi,100)
            
        # make new triangulation and contour map

        # plot scatter
        plt.figure()
        plt.scatter(x,y)
        plt.gca().axis('equal')
        plt.show()
        
        figsize=[7,3]
        n_plots = 3
        if add_autoscale_plots:
            figsize = [14,3]
            n_plots = 6
        fig = plt.figure(figsize=figsize) 

        cbar_ticks = np.linspace(min_for_scale, max_for_scale, n_ticks)
        # cbar_ticks = None
        
        range_diff = max_for_scale-min_for_scale
        power = ceil(np.log10(range_diff))
        n_ticks = np.round(range_diff*10**(-power))
        n_ticks_more = n_ticks*2**(1-np.round(n_ticks/10))
        tick_spacing = 10**power/2**(1-np.round(n_ticks/10))
        # cbar_ticks = np.arange(min_for_scale, max_for_scale, tick_spacing)

        cmap=matplotlib.cm.get_cmap('autumn_r')

        pad = 0.15
        pad = 0.04
        
        ax = fig.add_subplot(1,n_plots,1)
        # print("I-field data", data_x)
        # overrides the previous attempt at auto scaling
        if scale_range is None:
            print("AUTOSCALING 5 TICKS")
            min_for_scale = None
            max_for_scale = None
            levels = 257
            cbar_ticks = [] # ticker.MaxNLocator(5)
            cbar_ticks = ticker.MaxNLocator(nbins=5, prune='both')
        else:
            levels = np.linspace(min_for_scale, max_for_scale, 257)

        if rotate_90:
            pc1 = ax.tricontourf(list(y), list(x), data_x, cmap=cmap,\
                levels=levels, vmin=min_for_scale,vmax=max_for_scale, extend='both')
        else:
            pc1 = ax.tricontourf(list(x), list(y), data_x, cmap=cmap,\
                levels=levels, vmin=min_for_scale,vmax=max_for_scale, extend='both')
        # plt.plot(r_line*np.cos(phi_line), r_line*np.sin(phi_line), color='k', zorder=2)
        #display_r = ax.transData.transform(max_r)
        #Circle((display_r/2,display_r/2),display_r/2,zorder=-1)
        if use_circle_path:
            circ = Circle((0, 0), max_r*1.003,zorder=4,facecolor=None,edgecolor='k',linewidth=2)
            ax.add_patch(circ)
        else:
            plt.plot(r_line*np.cos(phi_line), r_line*np.sin(phi_line), color='k', zorder=2)

        if draw_NA_circle is not None:
            # draw_NA_circle is the radius of the circle to draw (sin(theta) or rho)
            plt.plot(draw_NA_circle*np.cos(phi_line), draw_NA_circle*np.sin(phi_line), color='k', zorder=2, linestyle='--')    
            
        ax.set_aspect('equal')
        ax.axis('off')
        ax.autoscale(True)
        ax.set_title("X component intensity", pad=-1.5, fontsize=11)
        
        cb = fig.colorbar(pc1, ax=ax, fraction=0.04, pad=pad, ticks=cbar_ticks, format='%.01f',location="bottom")
        cb.update_ticks()
        # have to do this every time? makes no sense
        if scale_range is None:
            tick_locator = ticker.MaxNLocator(nbins=5, prune='both')
            cb.locator = tick_locator
            cb.update_ticks()

        ### y
        ax2 = fig.add_subplot(1,n_plots,2)
        if rotate_90:
            pc2 = ax2.tricontourf(list(y), list(x), data_y, cmap=cmap,\
                levels=levels, vmin=min_for_scale,vmax=max_for_scale, extend='both')
        else:
            pc2 = ax2.tricontourf(list(x), list(y), data_y, cmap=cmap,\
                levels=levels, vmin=min_for_scale,vmax=max_for_scale, extend='both')
        if use_circle_path:
            circ = Circle((0, 0), max_r*1.003,zorder=4,facecolor=None,edgecolor='k',linewidth=2)
            ax2.add_patch(circ)   
        else: 
            plt.plot(r_line*np.cos(phi_line), r_line*np.sin(phi_line), color='k',zorder=4)

        if draw_NA_circle is not None:
            # draw_NA_circle is the radius of the circle to draw (sin(theta) or rho)
            plt.plot(draw_NA_circle*np.cos(phi_line), draw_NA_circle*np.sin(phi_line), color='k', zorder=2, linestyle='--')  

        ax2.set_aspect('equal')
        ax2.axis('off')
        ax2.autoscale(True)
        ax2.set_title("Y component intensity", pad=-1.5, fontsize=11)

        cb2 = fig.colorbar(pc2, ax=ax2, fraction=0.04, pad=pad, ticks=cbar_ticks, format='%.01f',location="bottom")
        # have to do this every time? makes no sense
        if scale_range is None:
            tick_locator = ticker.MaxNLocator(nbins=5, prune='both')
            cb2.locator = tick_locator
            cb2.update_ticks()


        ### total
        ax3 = fig.add_subplot(1,n_plots,3)
        if rotate_90:
            pc3 = ax3.tricontourf(list(y), list(x), data_total, cmap=cmap,\
                levels=levels, vmin=min_for_scale,vmax=max_for_scale, extend='both')
        else:
            pc3 = ax3.tricontourf(list(x), list(y), data_total, cmap=cmap,\
                levels=levels, vmin=min_for_scale,vmax=max_for_scale, extend='both')
        if use_circle_path:
            circ = Circle((0, 0), max_r*1.003,zorder=5,facecolor=None,edgecolor='k',linewidth=2)
            ax3.add_patch(circ) 
        else:
            plt.plot(r_line*np.cos(phi_line), r_line*np.sin(phi_line), color='k',zorder=5)

        if draw_NA_circle is not None:
            # draw_NA_circle is the radius of the circle to draw (sin(theta) or rho)
            plt.plot(draw_NA_circle*np.cos(phi_line), draw_NA_circle*np.sin(phi_line), color='k', zorder=2, linestyle='--')        

        ax3.set_aspect('equal')
        ax3.axis('off')
        ax3.autoscale(True)
        ax3.set_title("Total intensity", pad=-1.5, fontsize=11)

        cb3 = fig.colorbar(pc3, ax=ax3, fraction=0.04, pad=pad, ticks=cbar_ticks, format='%.01f',location="bottom")
        if scale_range is None:
            tick_locator = ticker.MaxNLocator(nbins=5, prune='both')
            cb3.locator = tick_locator
            cb3.update_ticks()

        y = 1.001

        if add_autoscale_plots:
            fig = self.plot_autoscale(fig, draw_NA_circle=draw_NA_circle)

        fig.suptitle(title,wrap=True)#,y=y)

        fig.text(.5, -0.05, caption_text, ha='center', fontsize=13)

        fig.set_tight_layout(True)
        # fig.tight_layout()
        # plt.subplots_adjust(wspace=0, hspace=-10)
        # plt.subplots_adjust(right=0, bottom=0)
        #plt.autoscale(True)
        plt.show()
        self.figure = fig
        return fig, (pc1, pc2, pc3)

    def save(self, save_dir):
        save_dir = os.path.normpath(save_dir)
        dirname =  os.path.dirname(save_dir)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        print("Saving in %s" % save_dir)
        self.figure.savefig(save_dir, bbox_inches='tight')

    def plot_autoscale(self, fig, rotate_90=False, use_circle_path=False,
                    max_r_in=None,pad=0.04, cmap=None, font_sizes=[14,12,11],
                    draw_NA_circle=None):
        """TODO reuse functions more, tidy up.. it's a mess"""
        ax = fig.add_subplot(1,6,4)

        if cmap is None:
            #cmap=matplotlib.cm.get_cmap('autumn_r')
            cmap=matplotlib.colormaps['autumn_r']

        (x, y) = self.x, self.y
        (data_x, data_y, data_total) =  self.data_x,\
            self.data_y, self.data_total
        
        max_r = np.max([np.max(x), np.max(y), abs(np.min(x)), abs(np.min(y))])
        print("max_r_in", max_r_in, "max_r", max_r)

        if max_r_in is not None:
            max_r = max([max_r_in, max_r])

        # draw line instead
        r_line = np.array([max_r]*100)
        phi_line = np.linspace(0, 2*np.pi,100)

        min_for_scale = None
        max_for_scale = None
        levels = 257
        cbar_ticks = ticker.MaxNLocator(nbins=4, prune='both')


        if rotate_90:
            pc1 = ax.tricontourf(list(y), list(x), data_x, cmap=cmap,\
                levels=levels, vmin=min_for_scale,vmax=max_for_scale, extend='both')
        else:
            pc1 = ax.tricontourf(list(x), list(y), data_x, cmap=cmap,\
                levels=levels, vmin=min_for_scale,vmax=max_for_scale, extend='both')

        if use_circle_path:
            circ = Circle((0, 0), max_r*1.003,zorder=4,facecolor=None,edgecolor='k',linewidth=2)
            ax.add_patch(circ)
        else:
            plt.plot(r_line*np.cos(phi_line), r_line*np.sin(phi_line), color='k', zorder=2)

        if draw_NA_circle is not None:
            # draw_NA_circle is the radius of the circle to draw (sin(theta) or rho)
            plt.plot(draw_NA_circle*np.cos(phi_line), draw_NA_circle*np.sin(phi_line), color='k', zorder=2, linestyle='--')

        ax.set_aspect('equal')
        ax.axis('off')
        ax.autoscale(True)
        ax.set_title("X component intensity", pad=-1.5, fontsize=11)
        
        cb = fig.colorbar(pc1, ax=ax, fraction=0.04, pad=pad, ticks=cbar_ticks, format='%.02f',location="bottom")

        cb.update_ticks()
        # have to do this every time? makes no sense

        tick_locator = ticker.MaxNLocator(nbins=4, prune='both')
        cb.locator = tick_locator
        cb.update_ticks()

        ### y
        ax2 = fig.add_subplot(1,6,5)
        if rotate_90:
            pc2 = ax2.tricontourf(list(y), list(x), data_y, cmap=cmap,\
                levels=levels, vmin=min_for_scale,vmax=max_for_scale, extend='both')
        else:
            pc2 = ax2.tricontourf(list(x), list(y), data_y, cmap=cmap,\
                levels=levels, vmin=min_for_scale,vmax=max_for_scale, extend='both')
        if use_circle_path:
            circ = Circle((0, 0), max_r*1.003,zorder=4,facecolor=None,edgecolor='k',linewidth=2)
            ax2.add_patch(circ)   
        else: 
            plt.plot(r_line*np.cos(phi_line), r_line*np.sin(phi_line), color='k',zorder=4)

        if draw_NA_circle is not None:
            # draw_NA_circle is the radius of the circle to draw (sin(theta) or rho)
            plt.plot(draw_NA_circle*np.cos(phi_line), draw_NA_circle*np.sin(phi_line), color='k', zorder=2, linestyle='--')

        ax2.set_aspect('equal')
        ax2.axis('off')
        ax2.autoscale(True)
        ax2.set_title("Y component intensity", pad=-1.5, fontsize=11)

        cb2 = fig.colorbar(pc2, ax=ax2, fraction=0.04, pad=pad, ticks=cbar_ticks, format='%.02f',location="bottom")
        # have to do this every time? makes no sense
        tick_locator = ticker.MaxNLocator(nbins=4, prune='both')
        cb2.locator = tick_locator
        cb2.update_ticks()


        ### total
        ax3 = fig.add_subplot(1,6,6)
        if rotate_90:
            pc3 = ax3.tricontourf(list(y), list(x), data_total, cmap=cmap,\
                levels=levels, vmin=min_for_scale,vmax=max_for_scale, extend='both')
        else:
            pc3 = ax3.tricontourf(list(x), list(y), data_total, cmap=cmap,\
                levels=levels, vmin=min_for_scale,vmax=max_for_scale, extend='both')
        if use_circle_path:
            circ = Circle((0, 0), max_r*1.003,zorder=5,facecolor=None,edgecolor='k',linewidth=2)
            ax3.add_patch(circ) 
        else:
            plt.plot(r_line*np.cos(phi_line), r_line*np.sin(phi_line), color='k',zorder=5)

        if draw_NA_circle is not None:
            # draw_NA_circle is the radius of the circle to draw (sin(theta) or rho)
            plt.plot(draw_NA_circle*np.cos(phi_line), draw_NA_circle*np.sin(phi_line), color='k', zorder=2, linestyle='--')

        ax3.set_aspect('equal')
        ax3.axis('off')
        ax3.autoscale(True)
        ax3.set_title("Total intensity", pad=-1.5, fontsize=11)

        cb3 = fig.colorbar(pc3, ax=ax3, fraction=0.04, pad=pad, ticks=cbar_ticks, format='%.02f',location="bottom")
        vmin_ = cb3.vmin
        vmax_ = cb3.vmax
        print("cb3.vmax bf", cb3.vmax)
        if (vmax_-vmin_) < 1e-3:
            cb3.vmax += 1e-2

        tick_locator = ticker.MaxNLocator(nbins=4, prune='both')
        cb3.locator = tick_locator
        cb3.update_ticks()

        return fig

def heatmap_plot(x0, y0, data_x, data_y, title=""):
    if len(x0) < 4:
        print("Insufficient points to plot heatmap")
        return

    fig = plt.figure(figsize=[10,3]) 

    max_for_scale = np.max([np.max(data_x), np.max(data_y)])
    min_for_scale = np.min([np.min(data_x), np.min(data_y)])

    import matplotlib
    cbar_ticks = np.linspace(min_for_scale, max_for_scale, 11)
    #cmap=matplotlib.cm.get_cmap('autumn_r')
    cmap=matplotlib.colormaps['autumn_r']

    ax = fig.add_subplot(131)
    # print("I-field data",  trace_E_vec[:,0]**2)
    levels = np.linspace(min_for_scale, max_for_scale, 257)


    pc1 = ax.tricontourf(x0, y0, data_x, cmap=cmap,\
        levels=levels, vmin=min_for_scale,vmax=max_for_scale, extend='both')

    # plt.plot(r_line*np.cos(phi_line), r_line*np.sin(phi_line), color='k', zorder=2)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title("X component intensity")
    
    fig.colorbar(pc1, ax=ax, fraction=0.04, pad=0.15, ticks=cbar_ticks,location="bottom")
    
    ax2 = fig.add_subplot(132)
    levels = np.linspace(min_for_scale, max_for_scale, 257)

    pc2 = ax2.tricontourf(x0, y0, data_y, cmap=cmap,\
        levels=levels, vmin=min_for_scale,vmax=max_for_scale, extend='both')

    # plt.plot(r_line*np.cos(phi_line), r_line*np.sin(phi_line), color='k', zorder=2)
    ax2.set_aspect('equal')
    ax2.axis('off')
    ax2.set_title("Y component intensity")
    
    fig.colorbar(pc2, ax=ax2, fraction=0.04, pad=0.15, ticks=cbar_ticks,location="bottom")
    ax3 = fig.add_subplot(133)
    levels = np.linspace(min_for_scale, max_for_scale, 257)

    pc3 = ax3.tricontourf(x0, y0, data_y+data_x, cmap=cmap,\
        levels=levels, vmin=min_for_scale,vmax=max_for_scale, extend='both')

    # plt.plot(r_line*np.cos(phi_line), r_line*np.sin(phi_line), color='k', zorder=2)
    ax3.set_aspect('equal')
    ax3.axis('off')
    ax3.set_title("Y+X component intensity")
    
    fig.colorbar(pc3, ax=ax3, fraction=0.04, pad=0.15, ticks=cbar_ticks,location="bottom")

    fig.suptitle(title)

    