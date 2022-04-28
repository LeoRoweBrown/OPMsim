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

        max_for_scale = (np.max(data_sum))
        # print("type: ", type(data_sum))
        # print("type: ", type(max_for_scale))
        # print("angles:", self.polar_angles)
        # print(data_sum)
        fig = plt.figure(figsize=[10,4])

        #Create a polar projection
        ax1 = fig.add_subplot(131, projection=projection)
        # pc1 = ax1.pcolormesh(angles,pupil_r_range,data_x.T,\
        pc1 = ax1.pcolormesh(self.polar_angles, self.polar_radii, self.data_x,\
             shading='auto', vmin=0,vmax=max_for_scale)
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
            shading='auto', vmin=0,vmax=max_for_scale)

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
            shading='auto', vmin=0,vmax=max_for_scale)
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