from copy import deepcopy
from matplotlib.pyplot import grid
from matplotlib import pyplot as plt
import numpy as np
from numpy.ma.core import masked_greater
import optical_matrices
import scipy.interpolate as interp
import anisotropy
import graphics

class Detector():
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
        self.unstructured_x = []  # unstructured x points on polar plot (for triangulation)
        self.unstructured_y = []
        self.rays = None

        self.interpolated = False
        self.is_binning_detector = False
        self.Ix_integral = None  # these are not implemented yet
        self.Iy_integral = None
        self.I_total_integral = None

    def detect_rays(self, rays):
        """
        Populate detector with rays, do coord transforms as necessary if 
        pupil surface is curved (happens when final lens is an objective)
        """
        rays.remove_escaped_rays()

        print(str(rays.n_final-rays.n) + " escaped out of " + str(rays.n))

        self.ray_polar_radius = np.array([None]*rays.n_final)
        self.ray_phi = np.array([None]*rays.n_final)

        if rays.isMeridional:  # put back into non meridional basis
            rays.meridional_transform(inverse=True)

        if self.curved:  # curved pupil
            rays.update_history()  # save rays before
            # print("before rotate", rays.I_vec)
            rays.rotate_rays_local_basis()

        print("Is curved?", self.curved)
        self.ray_polar_radius, self.ray_phi = \
            self.get_final_polar_coords(rays, self.curved)

        # squeeze to avoid broadcasting and (N,N) arrays instead of N
        # print(rays.I_vec)
        self.Ix_raw = np.absolute(rays.I_vec[:,0].squeeze())**2
        self.Iy_raw = np.absolute(rays.I_vec[:,1].squeeze())**2

        self.Ix_area_scaled = self.Ix_raw * rays.area_scaling
        self.Iy_area_scaled = self.Iy_raw * rays.area_scaling

        print("min in raw (x):", np.min(self.Ix_raw))
        print("min in raw (y):", np.min(self.Iy_raw))

        print("min in scaling:", np.min(rays.area_scaling))

        self.ray_phi = np.asarray(self.ray_phi, dtype=np.float64)

        phi_1d = self.ray_phi.squeeze()
        self.unstructured_x = self.ray_polar_radius*np.cos(phi_1d)
        self.unstructured_y = self.ray_polar_radius*np.sin(phi_1d)

        self.rays = rays
        self.integrate_pupil()

        # np.set_printoptions(threshold=1000)

    def integrate_pupil(self):
        # print(self.ray_areas)
        self.Ix_integral = 0
        self.Iy_integral = 0
        rays = self.rays
        total_area = np.sum(rays.areas)
        # print("total area of elements (curved)", total_area)
        scaled_areas = rays.areas * rays.area_scaling
        total_area = np.sum(scaled_areas)
        # TODO: I think because of energy conversation of these "rays"
        # that energy does not need to be scaled - I-change from focussing
        # compensated by area on pupil, it only matters for plotting 
        # purposes
        self.Ix_integral = np.sum(self.Ix_raw)# * rays.area_scaling)
        self.Iy_integral = np.sum(self.Iy_raw)# * rays.area_scaling)

        # area check
        print("total_area", total_area)

        self.I_total_integral = self.Ix_integral + self.Iy_integral

    def plot_pupil(self, title, fill_zeroes=True):
        print("ray count", self.rays.n_final)
        if self.rays.n_final < 4:
            print("not enough points to plot pupil, skipping")
            return
        pupil = graphics.PupilPlotObject(
            self.unstructured_x, self.unstructured_y,
            self.Ix_area_scaled, self.Iy_area_scaled)
        pupil.plot(title, fill_zeroes=fill_zeroes)
        self.figure = pupil

    def get_final_polar_coords(self, rays, curved):
        """polar plot does not deal with negative radii, r=theta or rho"""
        phi = rays.phi
        if curved:
            r = abs(np.sin(rays.theta))
        else:
            r = rays.rho
        neg_rhos = rays.rho < 0  # replace negative ray height with phi += pi
        ## WOULD use modulo but it gets rid of 2pi, which we need because
        ## interpolation doesn't wrap around from 2pi to 0.
        phi[neg_rhos] = phi[neg_rhos] + np.pi
        mask = (phi > 2*np.pi)*1
        phi = phi - mask*2*np.pi
        if not curved:  # abs already takes care of this in curved case
            r = -r
        # general moduluo in case phi > 2pi
        mask = (phi > 2*np.pi)*1 
        phi = phi - mask*2*np.pi
        return r, phi