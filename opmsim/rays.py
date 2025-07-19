"""
Class representing bundle of rays, with a 2D matrices for k-vector and E-vectors
"""
import os
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
        self.n_initial = self.n
        self.n_final = self.n
        self.lda = lda  # wavelength
        self.basis = np.array([[1, 0, 0],  # current cartesian basis
                              [0, 1, 0],
                              [0, 0, 1]])

        # these are all updated/affected by transforms
        # use column stack for (N,3) shape, where N is number of rays, then expand dims for broadcasting.
        self.k_vec = np.column_stack((
            np.sin(theta_array) * np.cos(phi_array),
            np.sin(theta_array) * np.sin(phi_array),
            np.cos(theta_array)
        ))
        self.k_vec = np.expand_dims(self.k_vec, axis=-1)
        self.e_field = np.zeros_like(self.k_vec)
        self.e_field_pre = 0j
        self.phi = phi_array
        self.theta = theta_array
        self.rho = np.zeros_like(phi_array)  # cylindrical coordinate rho, radial distance of ray from optical axis
        self.rho_before_trace = None  # used to store the rho before tracing, sometimes useful
        self.initial_path_length = initial_path_length  # for calculating initial phase, TODO: not used maybe remove
        
        # NOTE: pos and pos_global are temp variables, while path_coords/path_coords_local stores each position in full
        self.pos = np.zeros((len(phi_array), 3, 1))  # position of ray in Cartesian coords in current basis
        # current position of ray in Cartesian coords in original basis (n_rays, 3, 1), 1 for broadcasting
        self.pos_global = np.zeros((len(phi_array), 3, 1))
        self.path_coords = np.zeros((len(phi_array), 3, 1))  # (global basis) coords for ray path (n_rays, 3, n_coords)
        self.path_coords_local = np.zeros((len(phi_array), 3, 1))  # local basis version of path_coords

        self.total_intensity_initial = np.zeros_like(self.e_field)

        self.optical_axis = 0  # todo: make 3 element vector
        self.is_meridional = False
        self.escaped = [False] * self.n  # mask used to indicate rays that are lost

        if area_elements is None:
            area_elements = np.ones(self.n)  # TODO place with area calculation of cap
        self.areas = area_elements  # area elements dA assoicated with each ray that build up the spherical surface
        self.area_scaling = np.ones(self.n)  # for scaling energy when flat and curved wavefronts
        self.emission_scaling = np.ones(1)

        self.transfer_matrix = np.tile(
            np.identity(3, dtype=np.complex128), (1, self.n, 1, 1))
        self.negative_kz = False  # e.g., if ray is reflected back by mirror
        self.ray_density = self.n / np.sum(area_elements)  # so values dont change with ray number

        # Collection efficiency metrics
        self.emission_efficiency = 1  # EE
        self.half_sphere_energy = 1  # energy emitted by source in 2pi steradians, TODO: move to dipole source instead?
        self.average_energy_times_NA = 1  # half_sphere_energy scaled by actual collection angle

        self.keep_history = keep_history  # actually overriden by trace_rays which decides this..

        self.ray_history = []

    def verify_dot_product(self):
        if np.any(np.abs(np.sum(self.k_vec * self.e_field, axis=2)) > 1e-9):
            raise Exception("Dot product of E and k not zero for all rays!",
                  np.sum(self.k_vec * self.e_field, axis=2))

    def save_debug_data(self, path, save_efield=False):
        os.makedirs(path, exist_ok=True)
        np.savetxt(os.path.join(path, "k_vec.csv"), self.k_vec)
        np.savetxt(os.path.join(path, "pos.csv"), self.pos)
        np.savetxt(os.path.join(path, "rho.csv"), self.rho)
        if save_efield:
            np.savetxt(os.path.join(path, "e_field.csv"), self.e_field)

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

    def propagate(self, path_distance):
        """
        Update ray position based on current k vector and path distance to propagate
        TODO: think about case where path_distance is not the same for all rays?
        """
        inverse_basis = np.linalg.inv(self.basis)  # for working out global position
        path_distance = np.atleast_1d(path_distance)
        # first do the rho multiplication, path_distance shouldn't be expanded in dims for broadcasting
        self.rho = self.rho + path_distance * np.sin(self.theta)
        # then broadcast for pos and kvec which are (n_rays, 3, 1) in size
        path_distance = path_distance.reshape(path_distance.shape[0], 1, 1)
        self.pos += self.k_vec * path_distance.reshape(path_distance.shape[0], 1, 1)
        self.pos_global += (inverse_basis @ self.k_vec) * path_distance
        self.path_coords = np.append(self.path_coords, (self.pos_global), axis=2)
        self.path_coords_local = np.append(self.path_coords_local, (self.pos), axis=2)
        # self.pos_global += (self.basis @ self.k_vec) * path_distance

    def change_basis(self, basis: np.ndarray, calculate_efield=False):
        """
        Method to perform a change of (Cartesian) basis. Used for a folding mirror for example.
        Basis is expressed in terms of the global Cartesian basis ((1, 0, 0), (0, 1, 0), (0, 0, 1)).
        Therefore, to change basis, basis is INVERTED to global basis, then new basis is applied.

        Args:
            rays (PolarRays): PolarRays object associated with the DipoleSource to undergo basis change
            basis (np.ndarray): 3x3 matrix, rows are basis vectors, expressed in the CURRENT basis.
            calculate_efield (bool, optional): Apply matrices to rays.e-field. Only done when e-field is
                calculated sequentially in tracing instead of at end in one operation. Defaults to False.
        """
        print("changing basis from", self.basis, "to", basis)
        to_global_basis = np.linalg.inv(self.basis)  # to recover original basis
        self.k_vec = basis @ to_global_basis @ self.k_vec
        if calculate_efield:
            self.e_field = basis @ to_global_basis @ self.e_field
        self.transfer_matrix = basis @ to_global_basis @ self.transfer_matrix
        self.basis = basis
        self.update_polar_angles()

    def update_polar_angles(self):
        self.theta = np.arccos(self.k_vec[:, 2]).flatten()
        self.phi = np.arctan2(self.k_vec[:, 1], self.k_vec[:, 0]).flatten()

        negative_theta = self.theta < 0  # mask to replace negative ray height with phi += pi
        self.theta[negative_theta] = -self.theta[negative_theta]
        self.phi[negative_theta] = (self.phi[negative_theta] + np.pi) % (2 * np.pi)

    def calculate_intensity(self, scaling=None, scale_by_density=True):
        """
        calculate field intensity on wavefront surface for the current
        rays object, scaled by photoselection (scaling), which depends ontthe dipole object
        """
        if scaling is None:
            scaling = self.emission_scaling  # scaling from dipole photoselection
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
        if escaped is None:
            not_escaped = np.invert(self.escaped)

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
        self.pos = self.pos[not_escaped]
        self.pos_global = self.pos_global[not_escaped]
        self.path_coords = self.path_coords[not_escaped, :, :]
        self.path_coords_local = self.path_coords_local[not_escaped, :, :]

    def set_zero_escaped_rays(self, escaped=None):
        if escaped is None:
            escaped = self.escaped
        self.e_field[:, escaped, :, :] = 0
