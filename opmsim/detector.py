from copy import deepcopy
from matplotlib.pyplot import grid
from matplotlib import pyplot as plt
import numpy as np
from numpy.ma.core import masked_greater
from . import optical_matrices
import scipy.interpolate as interp
from . import anisotropy
from .tools import graphics

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
        self.max_r = None
        self.info = None

    def detect_rays(self, rays):
        """
        Populate detector with rays, do coord transforms as necessary if 
        pupil surface is curved (happens when final lens is an objective)
        """

        print(str(rays.n-rays.n_final) + " escaped out of " + str(rays.n))
        self.n_rays_initial = rays.n
        self.ray_polar_radius = np.array([None]*rays.n_final)
        self.ray_phi = np.array([None]*rays.n_final)

        self.ray_polar_radius, self.ray_phi = \
            self.get_final_polar_coords(rays, self.curved)

        x = self.ray_polar_radius*np.cos(self.ray_phi)
        y = self.ray_polar_radius*np.sin(self.ray_phi)

        # squeeze to avoid broadcasting and (N,N) arrays instead of N
        # print(rays.I_vec)
        self.Ix_raw = rays.I_total[:,0]
        self.Iy_raw = rays.I_total[:,1]

        self.Ix_area_scaled = self.Ix_raw.squeeze() * rays.area_scaling
        self.Iy_area_scaled = self.Iy_raw.squeeze() * rays.area_scaling

        # print(rays.area_scaling)

        self.ray_phi = np.asarray(self.ray_phi, dtype=np.float64)

        # plot radii to debug
        fig = plt.figure()
        plt.plot(self.ray_polar_radius)
        plt.title("ray radii on polar plot")
        plt.xlabel("ray number in list")
        plt.ylabel("radius")
        plt.show()

        phi_1d = self.ray_phi.squeeze()
        self.unstructured_x = self.ray_polar_radius*np.cos(phi_1d)
        self.unstructured_y = self.ray_polar_radius*np.sin(phi_1d)

        self.rays = rays
        self.n_rays = rays.n_final
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
        # print("total_area", total_area)

        self.I_total_integral = self.Ix_integral + self.Iy_integral

    def plot_pupil(self, title="", show_prints=False, plot_arrow=None,
            fill_zeroes=False, scale_range=None, rotate_90=False, caption=True, max_r_in=None):
        print("ray count", self.rays.n_final)
        if self.rays.n_final < 4:
            print("not enough points to plot pupil, skipping")
            return
        if caption:
            caption_text = r'$Ix/Iy = %f$, $RCE = %f$' % (self.Iy_Ix_ratio, self.energy_ratio)
        else:
            caption_text = None
        if max_r_in is None:
            max_r_in = self.max_r
        pupil = graphics.PupilPlotObject(
            self.unstructured_x, self.unstructured_y,
            self.Ix_area_scaled, self.Iy_area_scaled)
        pupil.plot(title, show_prints, plot_arrow,
            fill_zeroes, scale_range, rotate_90, caption_text, max_r_in)
        # self.pupil = pupil
        return pupil

    def get_final_polar_coords(self, rays, curved):
        """polar plot does not deal with negative radii, r=theta or rho"""
        phi = rays.phi
        if curved:
            r = np.sin(rays.theta)
            negative_r = r < 0
        else:
            r = rays.rho
            negative_r = rays.rho < 0  # replace negative ray height with phi += pi
        ## WOULD use modulo but it gets rid of 2pi, which we need because
        ## interpolation doesn't wrap around from 2pi to 0.
        phi[negative_r] = phi[negative_r] + np.pi
        mask = (phi > 2*np.pi)*1
        phi = phi - mask*2*np.pi
        # if not curved:  # abs already takes care of this in curved case
        #     r = -r
        # general moduluo in case phi > 2pi
        mask = (phi > 2*np.pi)*1 
        phi = phi - mask*2*np.pi
        return r, phi