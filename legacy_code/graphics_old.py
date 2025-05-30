import numpy as np
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import os


class PupilPlotObject():
    def __init__(self, polar_r, polar_angles, data_x, data_y, curved):
        self.polar_radii = polar_r
        self.polar_angles = polar_angles
        self.data_x = data_x
        self.data_y = data_y
        self.data_total = data_x + data_y
        self.curved = curved  # is curved surface

    def plot(self, title, save_dir=None,
            file_name=None, show_prints=False, plot_arrow=None, projection="polar",
            unstructured_data=False):
        if unstructured_data:
            self._plot_unstructured(title, save_dir=None,
                file_name=None, show_prints=False, plot_arrow=None, projection="polar")
        else:
            self._plot(title, save_dir=None,
                file_name=None, show_prints=False, plot_arrow=None, projection="polar")
        
    def _plot_unstructured(self, title, save_dir=None,
                file_name=None, show_prints=False, plot_arrow=None, projection="polar"):
        """tricontourf for plotting, and a polar transformation"""

        dipole_alpha = None
        dipole_phi = None

        print("----------------------------------")

        max_range = (np.max(self.unstructured_data_total))


        
        (x, y) = self.unstructured_x, self.unstructured_y
        (data_x, data_y, data_total) =  self.unstructured_data_x,\
            self.unstructured_data_y, self.unstructured_data_total

        # add a point on the other side of x?
        """
        minx = np.min(x)
        maxx = np.max(x)
        if abs(minx) - abs(maxx) > 1e-6:
            if minx < 0:
                x = np.append(x, -minx)
                y = np.append(y, 0)
                data_x = np.append(data_x, 0)
                data_y = np.append(data_y, 0)
                data_total = np.append(data_total, 0)
        elif abs(maxx) - abs(minx) > 1e-6:
            if maxx > 0:
                x = np.append(x, -maxx)
                y = np.append(y, 0)
                data_x = np.append(data_x, 0)
                data_y = np.append(data_y, 0)
                data_total = np.append(data_total, 0)
        """

        minx = np.min(x)
        maxx = np.max(x)
        xrange = maxx - minx
        max_r = np.max([np.max(x), np.max(y), abs(np.min(x)), abs(np.min(y))]) 
        print("max rad", max_r)
        print(np.max(x))
        print(np.max(y))

        x_width = np.max(x) - np.min(x)
        width_ratio = x_width/(2*max_r)

        ## fill zeroes
        from scipy.spatial.distance import cdist
        # zip
        xyzip = np.array(list(zip(x,y)))
        dists = cdist(xyzip,xyzip)
        # replace zeros
        dists[dists == 0] = 1e9
        dist_mins = np.min(dists, 0)
        max_dist_neighbours = np.max(dist_mins)
        mean_dist_neighbours = np.mean(dist_mins)

        ## now create radius of zeroes
        r = np.linspace(0, max_r,100)
        phi = np.linspace(0, 2*np.pi,200)
        r_m, phi_m = np.meshgrid(r, phi)
        x_m = r_m*np.cos(phi_m)
        y_m = r_m*np.sin(phi_m)
        x_bg = x_m.flatten()
        y_bg = y_m.flatten()

        # remove points in the region
        xy_m_zip = np.array(list(zip(x_bg,y_bg)))
        dist_to_data = cdist(xyzip, xy_m_zip)
        print("dist mat shape", dist_to_data.shape)
        min_dist_to_data = np.min(dist_to_data, 0)
        mask_bg = min_dist_to_data > max_dist_neighbours
        bg_circle_x = x_bg[mask_bg]
        bg_circle_y = y_bg[mask_bg]
        bg_len = len(bg_circle_x)
        bg_vals = np.zeros(bg_len)

        # now add these points onto data
        x = np.append(x, bg_circle_x)
        y = np.append(y, bg_circle_y)
        data_x = np.append(data_x, bg_vals)
        data_y = np.append(data_y, bg_vals)
        data_total = np.append(data_total, bg_vals)

        # bg plot of zero data in a circle
        bg_r = np.linspace(0, max_r, 20) 
        bg_a = np.linspace(0, 2*np.pi, 20)
        bg_v = np.zeros((20,20))
        bg_rad, bg_angle = np.meshgrid(bg_r, bg_a)
        
        fig = plt.figure(figsize=[10,3]) 

        ax = fig.add_subplot(131)
        print("I-field data", data_x)
        pc1 = ax.tricontourf(list(x), list(y), data_x,\
            levels=256, vmin=0,vmax=max_range)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title("X component intensity")
        
        # new colorbar axis https://stackoverflow.com/questions/19679409/python-matplotlib-how-to-move-colorbar-without-resizing-the-heatmap 
        # pc1_ax = ax.get_position()
        # ax.set_position([pc1_ax.x0*1.05, pc1_ax.y0, pc1_ax.width, pc1_ax.height])
        """
        colorax = plt.axes([pc1_ax.x0*1.05 + pc1_ax.width * 1.05,\
            pc1_ax.y0, 0.01, pc1_ax.height])
        fig.colorbar(pc1, cax=colorax, fraction=0.045, pad=0.18)
        """
        # from mpl_toolkits.axes_grid1 import make_axes_locatable
        # different method
        # divider = make_axes_locatable(ax)
        # cax = divider.append_axes('right', size='5%', pad=0.15)
        # fig.colorbar(pc1, cax=cax, orientation='vertical')
        fig.colorbar(pc1, ax=ax, fraction=0.04, pad=0.15)


        ### y

        ax2 = fig.add_subplot(132)
        pc2 = ax2.tricontourf(list(x), list(y), data_y,\
            levels=256, vmin=0,vmax=max_range)
        ax2.set_aspect('equal')
        ax2.axis('off')
        ax2.set_title("Y component intensity")

        # pc2_ax = ax2.get_position()
        # ax2.set_position([pc2_ax.x0*1.05, pc2_ax.y0, pc2_ax.width, pc2_ax.height])
        # colorax2 = plt.axes([pc2_ax.x0*1.05 + pc2_ax.width * 1.05,\
        #     pc2_ax.y0, 0.01, pc2_ax.height])
        # fig.colorbar(pc2, cax=colorax2, fraction=0.045, pad=0.18)
        fig.colorbar(pc2, ax=ax2, fraction=0.04, pad=0.15)

        ### total

        ax3 = fig.add_subplot(133)
        pc3 = ax3.tricontourf(list(x), list(y), data_total,\
         levels=256, vmin=0,vmax=max_range)
        ax3.set_aspect('equal')
        ax3.axis('off')
        ax3.set_title("Total intensity")

        # pc3_ax = ax3.get_position()
        # ax3.set_position([pc3_ax.x0*1.05, pc3_ax.y0, pc3_ax.width, pc3_ax.height])
        # colorax3 = plt.axes([pc3_ax.x0*1.05 + pc3_ax.width * 1.05, pc3_ax.y0, 0.01, pc3_ax.height])
        # fig.colorbar(pc3, cax=colorax3, fraction=0.045, pad=0.18)
        fig.colorbar(pc3, ax=ax3, fraction=0.04, pad=0.15)
        # plt.subplots_adjust(wspace=0.45, hspace=None)
        fig.suptitle(title)

        fig.tight_layout()
        if save_dir is not None:
            save_dir = os.path.normpath(save_dir)
            print("Saving in %s" % save_dir)
            if not os.path.isdir(save_dir):
                print("No directory, %s, making it" % save_dir)
                os.makedirs(save_dir)
            full_save_path = os.path.join(save_dir, \
                file_name)
            fig.savefig(full_save_path, bbox_inches='tight')
        # else:
        #     plt.show()
        return fig, (pc1, pc2, pc3)


    def _plot(self, title, save_dir=None,
            file_name=None, show_prints=False, plot_arrow=None, projection="polar"):
        """Plot arrow is 2x1 array with the alpha and phi values of single dipole
        so if the source isn't a single dipole it shouldn't be used"""

        dipole_alpha = None
        dipole_phi = None

        # data_sum = (data_x*data_x + data_y*data_y)**0.5
        data_sum = self.data_x + self.data_y
        # print(data_y)
        print("----------------------------------")

        max_range = (np.max(data_sum))
        # print("type: ", type(data_sum))
        # print("type: ", type(max_range))
        # print("angles:", self.polar_angles)
        # print(data_sum)
        fig = plt.figure(figsize=[10,4])

        #Create a polar projection
        ax1 = fig.add_subplot(131, projection=projection)
        # pc1 = ax1.pcolormesh(angles,pupil_r_range,data_x.T,\
        pc1 = ax1.pcolormesh(self.polar_angles, self.polar_radii, self.data_x,\
             shading='auto', vmin=0,vmax=max_range)
        # pc1 = ax1.pcolormesh(self.polar_radii, self.polar_angles, self.data_x,\
        #     shading='auto')

        fig.colorbar(pc1, ax=ax1, fraction=0.045, pad=0.18)

        if plot_arrow is not None:
            dipole_alpha = plot_arrow[0]
            dipole_phi = plot_arrow[1]
            arrow_len = np.abs(0.5*np.cos(dipole_alpha))
            flip_phi = (dipole_phi+np.pi) % (2*np.pi)
            # p1 = patches.FancyArrowPatch((flip_phi, arrow_len), (dipole_phi, arrow_len), arrowstyle='<->',\
            #      mutation_scale=20)
            p1 = patches.FancyArrowPatch((flip_phi, arrow_len), (dipole_phi, arrow_len), arrowstyle='<->',\
                mutation_scale=20)
            ax1.add_patch(p1)

        ax1.set_title("X polarisation pupil intensity distribution")
        # fix lims

        #Create a polar projection
        ax2 = fig.add_subplot(132, projection=projection)

        pc2 = ax2.pcolormesh(self.polar_angles, self.polar_radii, self.data_y,\
            shading='auto', vmin=0,vmax=max_range)

        fig.colorbar(pc2, ax=ax2, fraction=0.045, pad=0.18)

        if plot_arrow is not None:
            arrow_len = np.abs(0.5*np.cos(dipole_alpha))
            flip_phi = (dipole_phi+np.pi) % (2*np.pi)
            # p1 = patches.FancyArrowPatch((flip_phi, arrow_len), (dipole_phi, arrow_len), arrowstyle='<->',\
            #      mutation_scale=20)
            p2 = patches.FancyArrowPatch((flip_phi, arrow_len), (dipole_phi, arrow_len), arrowstyle='<->',\
                mutation_scale=20)
            ax2.add_patch(p2)

        ax2.set_title("Y polarisation pupil intensity distribution")

        ax3 = fig.add_subplot(133, projection=projection)

        pc3 = ax3.pcolormesh(self.polar_angles, self.polar_radii, data_sum,\
            shading='auto', vmin=0,vmax=max_range)
        fig.colorbar(pc3, ax=ax3, fraction=0.045, pad=0.18)

        if plot_arrow is not None:
            arrow_len = np.abs(0.5*np.cos(dipole_alpha))
            flip_phi = (dipole_phi+np.pi) % (2*np.pi)
            # p1 = patches.FancyArrowPatch((flip_phi, arrow_len), (dipole_phi, arrow_len), arrowstyle='<->',\
            #      mutation_scale=20)
            p3 = patches.FancyArrowPatch((flip_phi, arrow_len), (dipole_phi, arrow_len), arrowstyle='<->',\
                mutation_scale=20)
            ax3.add_patch(p3)

        ax3.set_title("X+Y polarisation pupil intensity distribution")

        # ax2.set_ylim([0, 1])
        fig.suptitle(title)

        fig.tight_layout()
        if save_dir is not None:
            save_dir = os.path.normpath(save_dir)
            print("Saving in %s" % save_dir)
            if not os.path.isdir(save_dir):
                print("No directory, %s, making it" % save_dir)
                os.makedirs(save_dir)
            full_save_path = os.path.join(save_dir, \
                file_name)
            fig.savefig(full_save_path, bbox_inches='tight')
        # else:
        #     plt.show()
        return fig, (pc1, pc2, pc3)

class PupilPlotDetector(PupilPlotObject):
    """Same as PupilObject but takes detector in constructor"""
    def __init__(self, detector):
        max_radius = detector.max_polar_radius  # if flat wavefront, rho, if curved, sine(theta)
        resolution = detector.resolution

        if not detector.interpolated:
            detector.interpolate_intensity()

        # unstructured data for tricontour
        self.unstructured_x = detector.unstructured_x
        self.unstructured_y = detector.unstructured_y
        self.unstructured_data_x = detector.Ix_raw
        self.unstructured_data_y = detector.Iy_raw
        self.unstructured_data_total = detector.Iy_raw + \
                detector.Ix_raw

        # binned data
        self.polar_radii = detector.ray_r_list
        self.polar_angles = detector.ray_phi_list
        self.data_x = detector.current_ifield_x 
        self.data_y = detector.current_ifield_y
        self.data_total = detector.current_ifield_x + detector.current_ifield_y

        self.curved = detector.curved

    def scale_intensity_from_curved_pupil(self):
        """ just for debugging, don't use this, TODO remove """
        # sin(theta)
        sin_theta = self.polar_radii
        # theta = np.arcsin(sin_theta)
        cos_theta = np.sqrt(1 - sin_theta*sin_theta)  # for scaling
        print(len(sin_theta))
        print(np.shape(self.data_x))
        print(len(cos_theta))
        for r_i in range(len(sin_theta)):
            self.data_x[r_i, :] /= cos_theta[r_i]
            self.data_y[r_i, :] /= cos_theta[r_i]