from typing import no_type_check

import Ray from ray
import numpy as np

class dipole:
    def __init__(
        self, phi, theta, lda_em=None, lda_exc=None) -> None:
        """dipole class to simulate a dipole emitter
        Arguments:
            phi <float> -- azimuthal angle, rotation about z axis [rad]
                           clockwise when looking along positive z (into paper)
            theta <float> -- angle of dipole from x axis (note that ray calculations
                     use a different convention - measuring theta from z) [rad].
            lda_em <float> -- emission of fluorophore/dipole 
                              (to be implemented) [nm]
            lda_exc <float> -- excitaiton of fluorophore/dipole (used)
                       for probability in photoselection? (to be implemented) [nm]
        Methods:

        """
        if lda_exc is None:
            lda_exc = 500e-9  # should be a distribution at some point

        self.theta = (theta + np.pi/2) % (2*np.pi)  # 'theta' for rays (from x axis)
        self.theta_dipole_coords = theta  # 'theta' definition for dipole (from y axis)
        self.phi = phi
        self.lda_em = lda_em
        self.lda_exc = lda_exc

        # get dipole angle
        # angles given with theta measured from x not z so cos(theta) <-> sin(theta)
        self.p_vec = [ np.cos(theta)*np.cos(phi), np.cos(theta)*np.sin(phi), \
            np.sin(theta)]
    
    def getEfield(self, theta, phi, r) -> np.ndarray:
        """propagate the E-field along r for an angle (theta_p, phi_p) in
           the pupil coordinates (?)
        """
        # E = (e^ikr/r)k^2(n x p) x n
        # n = [sin(theta_p)cos(phi) i, sin(theta_p)sin(phi_p) j, cos(theta_p) k]
        n_vec = [ np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), \
            np.cos(theta)]
        n_x_p = np.cross(n_vec,self.p_vec)
        k = 2*np.pi/self.lda_exc 
 
        E_vec = np.cross(n_x_p, n_vec)
        E_mag = (np.e**(1j*k*r)/r)*k**2  # replace with distribution of k (lambda_exc)
        return (E_vec, E_mag)
    
    def new_ray(self, theta, phi, r):
        """calculate new E-field based on propagation vector r_vec"""
        if not self.ray.exists():
            # use getEfield to calculate the E field based on a propagation vector
            # and the dipole distribution, i.e. this is used for the calculation
            # between the dipole and entrance pupil, then standard ray tracing is
            # used after that. In reality since calculating the actual entrance
            # pupil might be hard, I might just completely overfill the pupil
            # and lose rays along the way (not very efficient but hey)
            E_vec, E_mag = getEfield(theta, phi)
            self.ray.E_vec = getEfield(theta)
            self.ray.E_mag = E_mag  # prefactor
            self.ray.k_vec = [ 
                np.sin(theta)*np.cos(phi),\
                np.sin(theta)*np.sin(phi),\
                np.cos(theta)
            ]
            ## now get polarisation: E vector relative to k, only azimuthal
            # convert E and k to useful things for the ray
            # i.e. do Rz and Ry coord transforms, if they're correct, Ez should be
            # zero in the new coords
            

            ray = Ray(self.lda_exc, polarisation, direction, magnitude)

    def generate_pupil_rays(self):
        pass
    

class complex_dipole(dipole):
    """Includes beta, time correlation/lifetime, spectral fluoresence"""
    def __init__(
        self, phi, theta, lda_em=None, lda_exc=None, beta=0, lifetime=None,
        correlation=None) -> None:
        """dipole class to simulate a dipole emitter
        Arguments:
            phi <float> -- azimuthal angle, rotation about z axis [rad]
                           clockwise when looking along positive z (into paper)
            theta <float> -- angle of dipole from x axis (note that ray calculations
                     use a different convention - measuring theta from z) [rad].
            lda_em <float> -- emission of fluorophore/dipole 
                              (to be implemented) [nm]
            lda_exc <float> -- excitaiton of fluorophore/dipole (used)
                       for probability in photoselection? (to be implemented) [nm]
        Methods:
            getEfield - get E field at entrance pupil for given ray from dipole.
                        Run this before doing the ray tracing (from the pupil)
            depolarise - calculate the depolarisation of the dipole based on diffusion
        """
        if lda_exc is None:
            lda_exc = 500e-9
        self.theta_exc = self.theta
        # for if emission dipole is no parallel with excitation dipole

        if correlation is None:
            self.correlation = 1
        else
            self.correlation = correlation
        if lifetime is None:
            self.lifetime = 0
        self.lifetime = lifetime  # fluorescence lifetime
        self.beta = beta

        # dipole coords, are more intuitive to work with for the dipole angle imo
        # theta_dipole_coords=0 is perpendicular to optical axis z
        # theta_=0 is parallel to optical axis z

        # non parallel exc and em dipoles
        theta = theta + beta

        self.theta_dipole_coords = theta  # 'theta' definition for dipole (from y axis)
        # calculate new theta after diffusion #
        self.depolarise()

        # now convert these to the system coordinates
        self.theta = (theta + np.pi/2) % (2*np.pi)  # 'theta' for rays (from z axis)
        
        self.phi = phi
        self.lda_em = lda_em
        self.lda_exc = lda_exc

        # get dipole angle
        # angles given with theta measured from x not z so cos(theta) <-> sin(theta)
        self.p_vec = [ np.cos(theta)*np.cos(phi), np.cos(theta)*np.sin(phi), \
            np.sin(theta)]
    
    def getEfield(self, theta_p, phi_p, r) -> np.ndarray:
        """propagate the E-field along r for an angle (theta_p, phi_p) in
           the pupil coordinates (?)
        """
        # E = (e^ikr/r)k^2(n x p) x n
        # n = [sin(theta_p)cos(phi) i, sin(theta_p)sin(phi_p) j, cos(theta_p) k]
        n_vec = [ np.sin(theta_p)*np.cos(phi_p), np.sin(theta_p)*np.sin(phi_p), \
            np.cos(theta_p)]
        n_x_p = np.cross(n_vec,self.p_vec)
        k = 2*np.pi/self.lda_exc 
 
        E_unitvec = np.cross(n_x_p, n_vec)
        E_mag = (np.e**(1j*k*r)/r)*k**2  # replace with distribution of k (lambda_exc)
        return (E_unitvec, E_mag)

    def depolarise(self, direction=None):
        """only considers rotation in one dimension (for now? okay who am I kidding)
        Arguments:
            direction <1 or -1> -- explicit choice of diffusion direction in theta
        """
        # use Perin equation (Lakowicz, principles of fluoroscopy - chapter 10)
        D_corr = (6 * self.correlation)**(-1)
        mean_squ_rotation = 2*D*self.lifetime

        # in the dipole coords, easier to work perpendicular to optical axis (x)
        theta = self.theta_dipole_coords
        # expand out mean square rotation diffusion:
        # msd = <(theta_t - theta)>^2 = <theta_t^2> + theta^2 - 2theta*<theta_t>
        # where theta_t is angle after diffusion, solve the quadratic
        a = 1
        b = -2*theta
        c = theta**2 - mean_squ_rotation

        theta_t_positive = -b + (b**2 - 4*a*c)/2*a
        theta_t_negative = -b - (b**2 - 4*a*c)/2*a
        if direction is None:
            direction = (-1+2*np.random.randint(2))
        else
            if direction != -1 and direction != 1:
            raise ValueError("Direction of diffusion must be -1 or 1")
        self.theta_dipole_coords = \
            (direction==1)*theta_t_positive + (direction==-1)*theta_t_negative
        




