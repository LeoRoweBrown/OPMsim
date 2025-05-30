import numpy as np

class Dipoles():
    def __init__(self, phi_d_array, alpha_d_array, lda_em=None, lda_exc=None) -> None:
        """dipole class to simulate a dipole emitter
        Note: everything is in radians, but the optical system is in degrees
        Is this confusing?
        Arguments:
            phi_d <float> -- azimuthal angle, rotation about z axis [rad]
                           clockwise when looking along positive z (into paper)
            alpha_d <float> -- angle of dipole from x axis [rad].
            lda_em <float> -- emission of fluorophore/dipole 
                              (to be implemented) [nm]
            lda_exc <float> -- excitaiton of fluorophore/dipole (used)
                       for probability in photoselection? (to be implemented) [nm]
        Methods:
            get_rays(self, NA)
        """
        if lda_exc is None:
            lda_exc = 500e-9  # should be a distribution at some point

        self.alpha_d = alpha_d_array
        self.phi_d = phi_d_array
        self.lda_em = lda_em
        self.lda_exc = lda_exc
        self.density = np.ones(len(alpha_d_array))  # scales quantity/density/fractional quanitity of dipole

        self.rays = None

        # get dipole angle
        # angles given with theta measured from x not z so cos(theta) <-> sin(theta)

        self.p_vec = np.column_stack(
            (np.cos(alpha_d_array)*np.cos(phi_d_array),
            np.cos(alpha_d_array)*np.sin(phi_d_array),
            np.sin(alpha_d_array)))

    def get_initial_e_field(self, rays):
        # use _get_e_field to calculate the E field based on a propagation vector
        # and the dipole distribution
        n_vec = rays.k_vec
        p_vec = self.p_vec
        r = rays.radius

        # prepare to broadcast, index=0: dipoles index=1:n_rays
        p_vec = np.reshape(p_vec, (p_vec.shape[0], 1, 3))
        n_vec = np.reshape(n_vec, (1, n_vec.shape[0], 3))

        n_x_p = np.cross(n_vec,self.p_vec)
        k = 2*np.pi/self.lda_exc 
 
        self.e_field = np.cross(n_x_p, n_vec)
        self.E_pre  = (np.e**(1j*k*r)/r)*k**2  # replace with distribution of k (lambda_exc)

        # self.e_field = self.e_field.astype(complex)