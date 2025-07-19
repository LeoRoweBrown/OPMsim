import numpy as np
from .tools import graphics
from .visualization import pupil_plot
from .rays import PolarRays


class Detector:
    """Like a photodetector, mapped to wavefront surface"""

    def __init__(self, curved=True, max_radius=None, resolution=None):
        """
        max_theta=None means auto-scale size to ray with largest rho
        resolution means number of sample points for azumithal and radial
        """
        self.max_polar_radius = max_radius  # if flat wavefront, rho, if curved, sine(theta)
        self.resolution = resolution
        self.Ix_raw = []
        self.Iy_raw = []
        self.Ix_area_scaled = []  # for plotting - sampling of rays becomes non-uniform
        self.Iy_area_scaled = []  # so need to rescale intensity for accurate plots
        self.curved = curved  # curved pupil surface - use sine mapping to radius
        self.ray_polar_radius = []  # these are every point
        self.ray_phi = []
        self.x = []  # unstructured x points on polar plot (for triangulation)
        self.y = []

        self.interpolated = False
        self.is_binning_detector = False
        self.Ix_integral = 0.  # these are not implemented yet
        self.Iy_integral = 0.
        self.I_total_integral = 0.
        self.max_r = 0.
        self.info = None
        self.isinitial = False

    def detect_rays(self, rays: PolarRays):
        """
        Populate detector with rays, do coord transforms as necessary if
        pupil surface is curved (happens when final lens is an objective)
        """

        # Important to remove lost rays (also due to NaNs) and calculate final intensity
        rays.remove_escaped_rays()
        rays.calculate_intensity()

        print(str(rays.n - rays.n_final) + " rays escaped out of " + str(rays.n))
        self.n_rays_initial = rays.n
        self.ray_polar_radius = np.array([None] * rays.n_final)
        self.ray_phi = np.array([None] * rays.n_final)

        self.ray_polar_radius, self.ray_phi = self.get_final_polar_coords(rays, self.curved)

        # squeeze to avoid broadcasting and (N,N) arrays instead of N
        self.Ix_raw = rays.intensity_per_dipole_vector[:, 0]
        self.Iy_raw = rays.intensity_per_dipole_vector[:, 1]
        self.Ix_area_scaled = self.Ix_raw.squeeze() * rays.area_scaling
        self.Iy_area_scaled = self.Iy_raw.squeeze() * rays.area_scaling

        self.ray_phi = np.asarray(self.ray_phi, dtype=np.float64)

        phi_1d = self.ray_phi.squeeze()
        self.x = self.ray_polar_radius * np.cos(phi_1d)
        self.y = self.ray_polar_radius * np.sin(phi_1d)

        # TODO REMOVE, REMOVE BELOW
        # if self.curved:
        #     self.x, self.y = rays.k_vec[:, 0].flatten(), rays.k_vec[:, 1].flatten()
        # else:
        #     self.x, self.y = rays.pos[:, 0].flatten(), rays.pos[:, 1].flatten()

        self.rays = rays
        self.n_rays = rays.n_final
        self.integrate_pupil()

    def integrate_pupil(self):
        # print(self.ray_areas)
        self.Ix_integral = 0
        self.Iy_integral = 0
        rays = self.rays
        scaled_areas = rays.areas * rays.area_scaling

        # Note, energy is conserved per area element, but not intensity
        # energy is integrated over area U = ∫∫ I dA
        self.Ix_integral = np.sum(self.Ix_raw)
        self.Iy_integral = np.sum(self.Iy_raw)

        self.I_total_integral = self.Ix_integral + self.Iy_integral

    def plot_exit_pupil(self):
        try:
            return pupil_plot.plot_pupil_intensity(
                x=self.x, y=self.y, data_x=self.Ix_area_scaled, data_y=self.Iy_area_scaled)
        except ValueError:
            print("Failed to plot pupil, data are:")
            print("Ix:")
            print(self.Ix_area_scaled)
            print("Iy")
            print(self.Iy_area_scaled)
            raise

    def get_final_polar_coords(self, rays, curved):
        """polar plot does not deal with negative radii, r=theta or rho"""
        phi = rays.phi
        if curved:
            r = np.sin(rays.theta)
            negative_r = r < 0  # mask to replace negative ray height with phi += pi
        else:
            r = rays.rho
            negative_r = rays.rho < 0  # mask to replace negative ray height with phi += pi
        ## WOULD use modulo but it gets rid of 2pi, which we need because
        ## interpolation doesn't wrap around from 2pi to 0.
        phi[negative_r] = phi[negative_r] + np.pi
        mask = (phi > (2 * np.pi)) * 1
        phi = phi - mask * 2 * np.pi  # move points phi + 2pi to phi where phi > 0

        return r, phi
