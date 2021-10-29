import numpy as np

class PolarRay:
    def __init__(self, lda, e_vec, k_vec, prefactor):
        """
        Arguments:
        lda <float> -- wavelength of ray
        direction <ndarray> -- ray direction [x y z] in terms of polar coords phi
                        (yz plane), theta (from z e.g. in xz plane if y=0)
        polarisation <ndarray> -- E field vector [x y z] relative to direction
        prefactor <float> -- includes 1/r dependence from source and phase 

        Methods:
        """
        self.lda = lda
        E_vec = 0
        self.E_vec = e_vec
        self.k_vec = k_vec
        # convert these into 

        # [cos(phi_ray), sin(phi_ray), 0]
        self.prefactor = prefactor
        self.ray_history = []

    def propagate(self, r):
        pass


class MerdinonalRay:
    """
    Ray class for ray tracing once initial E and k vectors are passed,
    these are calculated by getEfield in the dipole class - the idea is to
    evaluate the field at the entrance pupil and trace from there
    
    MeridonalRay is the one used for ray tracing, might use PolarRay for
    mirror reflections, or might just do all the conversions interally -
    haven't decided.
    """
    def __init__(self, lda, polarisation, theta, phi, magnitude, prefactor=1):
        """
        Arguments:
        lda <float> -- wavelength of ray
        direction <ndarray> -- ray direction [x y z] in terms of polar coords phi
                           (yz plane), theta (from z e.g. in xz plane if y=0)
        polarisation <ndarray> -- E field vector [x y z] relative to direction
        prefactor <float> -- includes 1/r dependence from source and phase 

        Methods:
        """
        self.lda = lda

        # convert these into 

        # [cos(phi_ray), sin(phi_ray), 0]
        self.polarisation = polarisation  # only azimuthally polarisation considered
        self.theta = theta
        self.phi = phi
        self.magnitude = magnitude
        self.prefactor = prefactor
        # each time lens element is met, area scaling is multiplied by 
        # cos(theta_before)/cos(theta_after)
        self.area_scaling = 1  
        self.ray_history = []

    def propagate(self, r):
        pass
