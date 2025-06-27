# Base class for all elements

import numpy as np
from ..rays import PolarRays

class Element():
    def __init__(self, dz=0, element_type="", label=""):
        self.type = element_type
        self.update_history = False
        self.matrix_list = dict()
        self.dz = dz  # displacement relative to optical axis (defined as 3rd row in basis)
        self.coords = np.array([0, 0, 0])  # for placement when plotting (in global Cartesian coords)
        self.basis = np.array([[1, 0, 0],
                               [0, 1, 0],
                               [0, 0, 1]])
        self.use_previous_basis = True
        self.thickness = 0  # thickness of element in optical axis direction (for drawing). e.g. for lens it is f
        self.label = label

    def update_efield(self, rays: PolarRays):
        rays.e_field = rays.transfer_matrix @ rays.e_field

    def trace_rays(self, rays: PolarRays, calculate_efield=False, debug_dir=None):
        """
        Base method for tracing rays (PolarRays object).
        Implementation should apply transformations/tracing to the rays.k_vec, rays.theta, rays.phi, etc.,
        and also update the optical_matrix in rays.transfer_matrix. The rays object is mutated by this method.

        Note: currently not really used. Lenses have their own "trace_f" method applied in meridional
        plane, and this is important to the sign of "rho" which is used in the pupil plot.
        TODO: I am hoping to replace this with cartesian rays.pos, and calculate position in pupil from this
        independent of the polar coordinates rho, phi.
        CURRENTLY UNUSED. I think I will do ray propagation in PolarRays itself.

        Args:
            rays (np.ndarray): PolarRays object associated with the DipoleSource to be traced
            calculate_efield (bool, optional): Apply matrices to rays.e-field when rays are traced, 
                rather than e-fields only calculated at the end in one multiplication. Defaults to False.
            debug_dir (_type_, optional): directory to save k-vec (or e-field if calculate_field true)
                for debugging. Defaults to None.
        """

        rays.rho = rays.rho + self.dz * np.sin(rays.theta)
        rays.pos += rays.k_vec * self.dz

        raise NotImplementedError(f"trace_rays has not been implemented for the element {self.type}")
