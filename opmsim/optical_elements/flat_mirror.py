import os
from matplotlib import pyplot as plt
import numpy as np
from numpy.typing import NDArray
from .. import matrices
from ..rays import PolarRays
from .base_element import Element

class FlatMirror(Element):
    """
    Flat mirror with custom tilt about y axis (rot_y) and constant value for reflectance.
    See ProtectedMirror and UnprotectedMirror for thin film and Fresnel versions
    """

    def __init__(
            self,
            rot_y: float = 0.0,
            reflectance: float = 1.0,
            plot_debug=False,
            label=''):
        """
        Args:
            rot_y (float, optional): rotation about y-axis in degrees. Defaults to 0.0.
            reflectance (float, optional): constant reflectance (reflection matrix diagonals). Defaults to 1.0.
            plot_debug (bool, optional): CURRENTLY UNUSED TODO REMOVE. Defaults to False.
            label (str, optional): Element label. Defaults to ''.
        """

        super().__init__(
            element_type='FlatMirror',
            label=label)
        self.mirror_type = "perfect"  # e.g. uncoated, protected
        self.rot_y = rot_y  # rotation in y, in degrees
        if self.rot_y > 90:
            raise ValueError("Mirror rotation cannot exceed 90 degrees")
        self.reflectance = reflectance
        self.retardance = True
        self.basis = np.array([
            [-1, 0, 0],
            [0, 1, 0],
            [0, 0, -1]
        ])
        self.use_previous_basis = False  # important to make sure new basis is used!
        self.plot_debug = plot_debug

    # TODO: do we really care much about this typing stuff
    def calculate_fresnel_matrix(self, theta_i, wavelength) -> NDArray[np.complexfloating]:
        return np.identity(3, dtype=np.complexfloating) * self.reflectance

    def normalize(self, v, axis=1):
        norm = np.linalg.norm(v, axis=axis).reshape(v.shape[0], 1, 1)
        norm[norm == 0] = 1
        return v / norm

    def trace_rays(self, rays: PolarRays, calculate_efield=False, debug_dir=None):

        if self.update_history:
            rays.update_history()

        k_vec_norm = rays.k_vec / np.linalg.norm(rays.k_vec, axis=1).reshape(rays.k_vec.shape[0], 1, 1)

        # get N vector
        N = np.array([-np.tan(self.rot_y * np.pi / 180), 0, -1])
        N = N / np.linalg.norm(N)
        N = N.reshape(1, 3, 1)
        print("normal vector", N)

        p = np.cross(k_vec_norm, N, axis=1)  # get p vector (k × N) (s wave comp unit?)
        kdotN = np.sum(k_vec_norm * N, 1)
        r = np.cross(k_vec_norm, p, axis=1)  # get r vector (k × p) (p wave comp unit?)

        # normalize since we compute the angles without the normalization factor...
        p = self.normalize(p)
        r = self.normalize(r)

        ax = plt.figure().add_subplot(projection='3d')
        ax.quiver(0, 0, 0, N[0, 0, 0], N[0, 1, 0], N[0, 2, 0], color='red', label='N')
        ax.quiver(0, 0, 0, p[0, 0, 0], p[0, 1, 0], p[0, 2, 0], color='blue', label='p')
        ax.quiver(0, 0, 0, r[0, 0, 0], r[0, 1, 0], r[0, 2, 0], color='green', label='r')
        ax.legend()

        parallel = r[:, :, 0]
        senkrecht = p[:, :, 0]

        ps_project = matrices.transformation.ps_projection_matrix(
            parallel, senkrecht, np.squeeze(rays.k_vec))

        inv_ps_proj = np.linalg.inv(ps_project)

        # get angle
        kxN = np.cross(k_vec_norm, N, axis=1)
        sin_mr_1theta = np.linalg.norm(abs(kxN), axis=1)
        theta_i = np.arcsin(sin_mr_1theta)

        M_fresnel = self.calculate_fresnel_matrix(theta_i=theta_i, wavelength=rays.lda)

        if not self.retardance:  # ignore retardance (absolute reflectivity)
            print("IGNORING IMAGINARY PARTS IN REFLECTANCE I.E. RETARDANCE/POLARISATION CHANGE")
            M_fresnel = np.absolute(M_fresnel)

        # Householder reflection matrix
        reflection_mat = matrices.transformation.reflection_cartesian_matrix(N.squeeze())

        rays.transfer_matrix = reflection_mat @ inv_ps_proj @ M_fresnel @ ps_project @ rays.transfer_matrix
        k_vec_ref = reflection_mat @ rays.k_vec
        rays.k_vec = k_vec_ref

        # Update basis for new optic axis
        basis = np.array([
            [1, 0, 0],
            [0, -1, 0],
            [0, 0, -1]
        ])

        rays.change_basis(basis)

class ProtectedFlatMirror(FlatMirror):
    """Inherits from FlatMirror, calculates Fresnel matrix from thin-film theory for protected mirror"""
    def __init__(
            self,
            rot_y: float = 0,
            film_thickness: float = 100.e-9,
            n_film_file: str = "../refractive_index_data/SiO2.txt",
            n_substrate_file: str = "../refractive_index_data/Ag.txt",
            retardance=True,
            label=""):

        super().__init__(
            rot_y=rot_y,
            reflectance=1,
            label=label)

        self.type = "ProtectedFlatMirror"
        self.mirror_type = "protected"  # e.g. perfect, uncoated, protected
        self.rot_y = rot_y  # rotation in y
        self.single_surface = True
        self.optical_layers = {}  # TODO, make dict to store n_data and thickness etc.

        self.n_film_file = n_film_file
        self.n_film_data = np.genfromtxt(self.n_film_file, delimiter="\t")
        self.n_film_data = self.n_film_data[1:, :]  # remove headers  # TODO, better parsing
        self.film_thickness = film_thickness

        self.n_substrate_file = n_substrate_file
        self.n_substrate_data = np.genfromtxt(self.n_substrate_file, delimiter="\t")
        self.n_substrate_data = self.n_substrate_data[1:, :]

        self.retardance = retardance

    def calculate_fresnel_matrix(self, theta_i, wavelength):
        matrix, _ = matrices.fresnel.thin_film_fresnel_matrix(
            theta_i, self.n_film_data,
            self.film_thickness, self.n_substrate_data,
            wavelength)
        return matrix

class UncoatedFlatMirror(FlatMirror):
    def __init__(
            self,
            rot_y: float = 0.,
            n_file: str = "../refractive_index_data/SiO2.txt",
            retardance=True,
            label=""):

        super().__init__(
            rot_y=rot_y,
            label=label)

        self.n_file = n_file
        self.n_data = np.genfromtxt(self.n_file, delimiter="\t")
        self.n_data = self.n_data[1:, :]  # remove headers  # TODO, better parsing, pandas?
        self.retardance = retardance

    def calcuate_fresnel_matrix(self, theta_i, wavelength):
        return matrices.fresnel.single_surface_fresnel_matrix(
            theta_i, self.n_data, wavelength)
