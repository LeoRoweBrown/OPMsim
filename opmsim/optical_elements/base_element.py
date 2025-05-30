# Base class for all elements

import numpy as np

class Element():
    def __init__(self, dz=None):
        self.type = 'Empty element'
        self.update_history = False
        self.matrix_list = dict()
        self.dz = dz

    def trace_rays(self, rays):
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
            rays (np.ndarray): PolarRays object associated with the DipoleSource
        """

        rays.rho = rays.rho + self.dz * np.sin(rays.theta)
        rays.pos += rays.k_vec * self.dz

        raise NotImplementedError(f"trace_rays has not been implemented for the element {self.type}")
