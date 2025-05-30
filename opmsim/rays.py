"""
Class representing bundle of rays, with a 2D matrices for k-vector and E-vectors
"""

from copy import deepcopy
import warnings
from math import ceil
import numpy as np
import copy
import matplotlib.pyplot as plt

# vectorised version of ray
class PolarRays:
    """
    Class representing a collection of rays, so that calculations are efficient and vectorized
    """
    def __init__(
            self, phi_array, theta_array, initial_path_length=None,
            area_elements=None, lda=500e-9, keep_history=True):
        """
        Args:
            phi_array (np.ndarray): array of ray azimuthal angle (measured from x)
            theta_array (np.ndarray): array of ray polar angle (measured from z)
            initial_path_length (np.ndarray): initial ray path length.
                Equals focal distance for curved principal surface
            area_elements (np.ndarray, optional): area element associated with each ray when generated in sphere.
                Defaults to None, and areas are caluclated based on cap area divided by ray count
            lda (float, optional): wavelength (lda is short for lambda). Defaults to 500e-9.
            keep_history (bool, optional): save the state of the object if True. Defaults to True.
        """
        self.n = len(phi_array)
        self.n_final = self.n
        self.lda = lda  # wavelength

        # these are all updated/affected by transforms
        # use column stack for (N,3) shape, where N is number of rays, then expand dims for broadcasting.
        self.k_vec = np.column_stack((
            np.sin(theta_array) * np.cos(phi_array),
            np.sin(theta_array) * np.sin(phi_array),
            np.cos(theta_array)
        ))
        self.k_vec = np.expand_dims(self.k_vec, axis=-1)
        self.e_field = np.zeros_like(self.k_vec)
        self.phi = phi_array
        self.theta = theta_array
        self.rho = np.zeros_like(phi_array)  # cylindrical coordinate rho, radial distance of ray from optical axis
        self.rho_before_trace = None  # used to store the rho before tracing, sometimes useful
        self.initial_path_length = initial_path_length  # for calculating initial phase, TODO: not used maybe remove
        self.pos = np.zeros((len(phi_array), 3, 1))  # position of ray in Cartesian coords
        self.total_intensity_initial = np.zeros_like(self.e_field)

        self.optical_axis = 0  # todo: make 3 element vector
        self.is_meridional = False
        self.escaped = [False] * self.n  # mask used to indicate rays that are lost

        if area_elements is None:
            area_elements = np.ones(self.n)  # TODO place with area calculation of cap
        self.areas = area_elements  # area elements dA assoicated with each ray that build up the spherical surface
        self.transfer_matrix = np.tile(np.identity(3), (1, self.n, 1, 1))  # TODO move to somewhere else?
        self.negative_kz = False  # e.g., if ray is reflected back by mirror
        self.ray_density = self.n / np.sum(area_elements)  # so values dont change with ray number

        # Collection efficiency metrics
        self.emission_efficiency = 1  # EE
        self.half_sphere_energy = 1  # energy emitted by source in 2pi steradians, TODO: move to dipole source instead?
        self.average_energy_times_NA = 1  # half_sphere_energy scaled by actual collection angle

        self.keep_history = keep_history  # actually overriden by trace_rays which decides this..

        self.ray_history = []

    def update_history(self, label=None):
        """
        Make a copy of the rays object and store in history: very RAM inefficient.
        Todo: either remove or make the history save to disk
        """
        self.label = label
        if self.keep_history:
            warnings.warn("Update history being reimplemented to save to disk!")
        else:
            # no reason to really have this option
            print("History has been disabled, rays not saved!")

    def propagate_rays(self, path_distance):
        """
        Update ray position based on current k vector and path distance to propagate
        TODO: think about case where path_distance is not the same for all rays?
        """
        self.pos += self.k_vec * path_distance

    def get_intensity(self, scaling=np.array([1]), scale_by_density=True):
        """
        calculate field intensity on wavefront surface for the current
        rays object, scaled by photoselection (scaling), which depends ontthe dipole object
        """
        intensity_vector = np.real(self.e_field * np.conj(self.e_field)) * scaling
        self.intensity_per_dipole_vector = np.mean(intensity_vector, axis=0)
        intensity_per_dipole = np.sum(self.intensity_per_dipole_vector, axis=1)
        total_intensity = np.sum(intensity_per_dipole)
        # I tried scaling by "ray density" so the answers are independent
        # of ray sampling, but ray number/solid angle appears to vary slightly
        # so this is disabled for now
        if scale_by_density:
            ray_density = self.ray_density
        else:
            ray_density = 1
        self.intensity_vector = intensity_vector
        # total_intensity_normalized
        self.total_intensity_normalized = total_intensity / ray_density
        self.total_power = intensity_per_dipole * self.areas  # getting the SI units right

    def remove_escaped_rays(self, escaped=None):
        """
        Remove rays that are lost in the tracing, makes simulation more efficient
        is there a more efficient way of doing this?
        """
        lost = deepcopy(self)
        if escaped is None:
            # print(self.escaped)
            not_escaped = np.invert(self.escaped)
            # print(not_escaped)

            escaped = self.escaped
        else:  # supply different escaped array that from self
            not_escaped = np.invert(escaped)
        self.e_field = self.e_field[:, not_escaped, :, :]
        self.k_vec = self.k_vec[not_escaped, :, :]
        self.transfer_matrix = self.transfer_matrix[:, not_escaped, :, :]
        self.phi = self.phi[not_escaped]
        self.theta = self.theta[not_escaped]
        self.rho = self.rho[not_escaped]
        self.area_scaling = self.area_scaling[not_escaped]
        self.areas = self.areas[not_escaped]

    def combine_rays(self, rays2):
        self.e_field = np.append(self.e_field, rays2.e_field, axis=1)
        self.k_vec = np.append(self.k_vec, rays2.k_vec, axis=0)
        self.transfer_matrix = \
            np.append(self.transfer_matrix, self.transfer_matrix, axis=0)
        self.phi = np.append(self.phi, rays2.phi)
        self.theta = np.append(self.theta, rays2.theta)
        self.rho = np.append(self.rho, rays2.rho)
        self.area_scaling = np.append(self.area_scaling, rays2.area_scaling)
        self.areas = np.append(self.areas, rays2.areas)

    def set_zero_escaped_rays(self, escaped=None):
        if escaped is None:
            escaped = self.escaped
        self.e_field[:, escaped, :, :] = 0
