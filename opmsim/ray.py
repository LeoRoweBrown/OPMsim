import numpy as np
import copy

class PolarRay:
    def __init__(self, phi, theta, rho, I_vec, k_vec, area, lda=500e-9 keep_history=True):
        """
        Arguments:
        lda <float> -- wavelength of ray
        direction <ndarray> -- ray direction [x y z] in terms of polar coords phi
                        (yz plane), theta (from z e.g. in xz plane if y=0)
        polarisation <ndarray> -- E field vector [x y z] relative to direction
        prefactor <float> -- includes 1/r dependence from source and phase 

        Methods:
        """
        self.lda = lda  # wavelength
        # these are all updated/affected by transforms
        self.I_vec = I_vec
        self.k_vec = k_vec
        # the following are contained in k_vec info?
        self.phi = phi
        self.theta_in = theta  # NOT UPDATED
        self.theta = theta
        self.rho = rho  # allow to be negative for correct refraction
        self.z = 0  # maybe unused, who knows, distance along optic axis
        self.isMeridional = False
        # if ray is above (+1) or below (-1) optic axis
        self.orientation = 1
        self.escaped = False
        self.area_scaling = 1

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

    def cartesian_to_polar(self, vector):
        # takes k_vec/I_vec and obtains phi and theta, phi used in transform
        # unused?
        vector_theta = np.arccos(vector[2])
        if vector[0] < 1e-5:
            vector_phi = vector[1]/abs(vector[1])  # get the sign right
        vector_phi = np.arctan2(vector[1], vector[0])
        vector_norm = vector_phi 

    def meridonial_transform(self):
        merid_transform = np.array(
            [[np.cos(self.phi), np.sin(self.phi), 0],
            [-np.sin(self.phi), np.cos(self.phi), 0],
            [0, 0, 1]]
        )
        self.I_vec = np.matmul(optical_matrices.meridional_transform(self.phi), self.I_vec)
        self.k_vec = np.matmul(optical_matrices.meridional_transform(self.phi), self.k_vec)
        self.isMeridional = True