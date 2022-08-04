from re import M
import numpy as np
from rays import PolarRays
import optical_matrices

class Dipole():
    def __init__(self, phi_d, alpha_d, lda_em=None, lda_exc=None) -> None:
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

        self.alpha_d = alpha_d
        self.phi_d = phi_d
        self.lda_em = lda_em
        self.lda_exc = lda_exc
        self.density = 1  # scales quantity/density/fractional quanitity of dipole

        self.rays = None

        # get dipole angle
        # angles given with theta measured from x not z so cos(theta) <-> sin(theta)
        self.p_vec = [ np.cos(alpha_d)*np.cos(phi_d),\
            np.cos(alpha_d)*np.sin(phi_d), \
            np.sin(alpha_d)]

    def _get_efield_tensor(self, phi, theta, r) -> np.ndarray:
        """propagate the E-field along r for an angle (theta, phi) in
           the curved pupil coordinates
        Arguments:
            phi array<float> -- azimuthal angle, rotation about z axis [rad]
                           clockwise when looking along positive z (into paper)
            theta array<float> -- polar angle measured from z axis [rad].
        Returns:
            (E_vec, E_mag, n_vec) -- Electric field vector, Electric field prefactor
                                     (depends on propagation distance and wavelength)
                                     and propagation direction
        """
        # E = (e^ikr/r)k^2(n x p) x n
        # n = [sin(theta_p)cos(phi) i, sin(theta_p)sin(phi_p) j, cos(theta_p) k]
        n_vec = np.column_stack((np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), \
            np.cos(theta)))
        n_x_p = np.cross(n_vec,self.p_vec)
        k = 2*np.pi/self.lda_exc 
 
        E_vec = np.cross(n_x_p, n_vec)
        E_mag = (np.e**(1j*k*r)/r)*k**2  # replace with distribution of k (lambda_exc)

        E_vec = E_vec.astype(complex)   # hope this doesnt slow things too much

        return (E_vec, E_mag, n_vec)

    def _get_efield(self, phi, theta, r) -> np.ndarray:
        n_rays = len(phi)
        E_vec = np.zeros((n_rays, 3))
        E_mag = np.zeros(n_rays, dtype=complex)
        n_vec = np.zeros((n_rays, 3))
        for n in range(len(phi)):
            n_vec[n, :] = np.array((np.sin(theta[n])*np.cos(phi[n]), np.sin(theta[n])*np.sin(phi[n]), \
                np.cos(theta[n])))
            n_x_p = np.cross(n_vec[n, :],self.p_vec)
            k = 2*np.pi/self.lda_exc 
    
            E_vec[n, :] = np.cross(n_x_p, n_vec[n, :])
            E_mag[n] = (np.e**(1j*k*r)/r)*k**2  # replace with distribution of k (lambda_exc)

            # n is directional vector, must be orthogonal to E
            dotp = np.sum(n_vec[n,:]*E_vec[n,:])
            if abs(dotp) > 1e-10:
                raise ValueError("E not orthogonal to k")
            # print("dot product E and k", dotp)
        return (E_vec, E_mag, n_vec)

    def generate_rays(self, r, phi_list, theta_list, ray_area_list):
        # use _get_efield to calculate the E field based on a propagation vector
        # and the dipole distribution
        n_rays = len(phi_list)
        phi_list = np.array(phi_list)
        theta_list = np.array(theta_list)
        ray_area_list = np.array(ray_area_list)

        E_vec, E_pre, k_vec = self._get_efield(phi_list, theta_list, r)
        
        mag_E_vec = np.reshape(np.linalg.norm(E_vec, axis=1), (n_rays, 1))
        I = E_vec #* mag_E_vec
        """
        # we don't actually use complex numbers anymore
        # rotate into E_z = 0 and square then put back into global basis
        basis_tensor = optical_matrices.rotate_basis_tensor(phi_list, theta_list)
        E_vec = basis_tensor @ np.reshape(E_vec, (n_rays,3,1))
        meridional = optical_matrices.meridional_transform_tensor(phi_list) # recover global xy
        invert_meridional = optical_matrices.meridional_transform_tensor(phi_list, inverse=True)
        E_vec = invert_meridional @ E_vec
        I = E_vec * np.absolute(E_vec)  # we add intensities (add field incoherently)
        I = meridional @ I  # back into meridional 
        # print("Iz after rotation", I[:,2])
        invert_basis_tensor = np.linalg.inv(basis_tensor)
        
        I = invert_basis_tensor @ I

        I = np.reshape(I, (n_rays,3))
        E_vec = np.reshape(E_vec, (n_rays,3))
        """

        # I = (E_vec * np.conjugate(E_vec)).real  # sum all intensities at first surface
        # zeros = np.abs(E_vec) < 1e-10
        # sign = np.sign(E_vec[~zeros])
        # I[~zeros] = I[~zeros]*sign # recover the sign/direction
        # I[zeros] = 0
        # I = I.real

        # check orthogonality again
        # I_k_dot = np.sum(I * k_vec, axis=1)
        # print("dot in generate rays", I_k_dot)
        # E_k_dot = np.sum(E_vec * k_vec, axis=1)
        # print("dot in generate rays (E-field)", E_k_dot)

        self.rays = PolarRays(phi_list, theta_list, I, k_vec, ray_area_list)

        ## now get polarisation: E vector relative to k, only azimuthal
        # Ez should be zero in the new coords defined by k