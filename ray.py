import numpy as np
import copy

class PolarRay:
    def __init__(self, lda, phi, theta, rho, e_vec, k_vec, prefactor, keep_history=True):
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
        # these are all updated/affected by transforms
        self.E_vec = e_vec
        self.k_vec = k_vec
        # the following are contained in k_vec info?
        self.phi = phi
        self.theta_in = theta  # NOT UPDATED
        self.theta = theta
        self.rho = rho  # allow to be negative for correct refraction
        self.z = 0  # maybe unused, who knows, distance along optic axis
        self.phase = 0
        self.isMeridional = False
        # if ray is above (+1) or below (-1) optic axis
        self.orientation = 1
        self.escaped = False
        self.area_scaling = 1
        # convert these into 

        # [cos(phi_ray), sin(phi_ray), 0]
        self.prefactor = prefactor

        self.keep_history = keep_history
        self.ray_history = []

    def get_theta(self, attr=False): # not used due to ambiguity of sign?
        # calc theta based on k_vec x,y,z
        if attr:
            return self.theta
        if self.k_vec[2]*1e-6 > \
            (self.k_vec[0]*self.k_vec[0] + self.k_vec[1]*self.k_vec[1]):
            return 0  # if |k_xy| = 0
        else:
            return np.arctan((self.k_vec[0]*self.k_vec[0] \
                + self.k_vec[1]*self.k_vec[1])**0.5/self.k_vec[2])

    def update_history(self, note=None):
        self.note = note
        if self.keep_history:
            self.ray_history.append(copy.deepcopy(self))

    def propagate(self, r):
        # to remove?
        pass

    def cartesian_to_polar(self, vector):
        # takes k_vec/E_vec and obtains phi and theta, phi used in transform
        vector_theta = np.arccos(vector[2])
        if vector[0] < 1e-5:
            vector_phi = vector[1]/abs(vector[1])  # get the sign right
        vector_phi = np.arctan(vector[1]/vector[0])
        vector_norm = vector_phi 
        pass

    def meridonial_transform(self):
        merid_transform = np.array(
            [[np.cos(self.phi), np.sin(self.phi), 0],
            [-np.sin(self.phi), np.cos(self.phi), 0],
            [0, 0, 1]]
        )
        # needs phi? obtain from k_vec?
        # change coords to medionial/sagittal
        pass

class MerdinonalRay:
    """
    Not really going to use this!!!
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
