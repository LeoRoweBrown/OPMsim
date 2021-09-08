import numpy as np

class Ray:
    """ray class for ray tracing once initial E and k vectors are passed,
    these are calculated by getEfield in the dipole class - the idea is to
    evaluate the field at the entrance pupil and trace from there"""
    def __init__(self, lda, polarisation, direction, magnitude, prefactor):
        """
        Arguments:
        lda <float> -- wavelength of ray
        direction <ndarray> -- ray direction [x y z] in terms of polar coords phi
                           (yz plane), theta (from z e.g. in xz plane if y=0)
        polarisation <ndarray> -- E field vector [x y z] relative to direction

        Methods:
        """
        self.lda = lda
        # self.E_vec = E_vec
        # self.k_vec = k_vec

        # convert these into 

        # [cos(phi_ray), sin(phi_ray), 0]
        self.polarisation = polarisation  # only azimuthally polarisation considered
        self.direction = direction
        self.magnitude = magnitude
        self.prefactor = prefactor
        self.ray_history = []

    def propagate(self, r):
        pass

