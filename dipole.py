from typing import no_type_check

import numpy as np
from ray import Ray
from tools import printif
from scipy.interpolate import griddata

class Dipole:
    def __init__(
        self, phi, theta, lda_em=None, lda_exc=None) -> None:
        """dipole class to simulate a dipole emitter
        Note: everything is in radians, but the optical system is in degrees
        Is this confusing?
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
    
    def getEfield_z(self, theta, phi, z=1, dx=0, dy=0, dz=0):
        """ PROBABLY WONT BE USED 
        propagate the E-field to z position at angle (theta, phi) in
        pupil coordinates, (not same theta as dipole) with dipole
        position (dx, dy, dz) from optical axis.
        z is distance along optical axis to pupil, e.g. working distance of objective.
        Arguments:
            phi <float> -- azimuthal angle, rotation about z axis [rad]
                           clockwise when looking along positive z (into paper)
            theta <float> -- polar angle measured from z axis [rad].
            z <float> (1.0) -- distance from source to pupil in z (along optical axis) [m]
            dx <float> (0.0) -- displacement of source from optical axis in x [m]
            dy <float> (0.0) -- displacement of source from optical axis in y [m]
            dz <float> (0.0) -- displacement of source from optical axis in z [m]
        Returns:
            (E_vec, E_mag, n_vec) -- Electric field vector, Electric field prefactor
                                     (depends on propagation distance and wavelength)
                                     and propagation direction
        """
        # E = (e^ikr/r)k^2(n x p) x n
        # n = [sin(theta_p)cos(phi) i, sin(theta_p)sin(phi_p) j, cos(theta_p) k]
        n_vec = [z*np.tan(theta)*np.cos(phi)-dx, z*np.tan(theta)*np.sin(phi)-dy, z-dz]
        n_x_p = np.cross(n_vec,self.p_vec)
        k = 2*np.pi/self.lda_exc 

        r = (n_vec[0]**2 + n_vec[1]**2 + n_vec[2]**2)**0.5
        n_vec /= r  # normalise
 
        E_vec = np.cross(n_x_p, n_vec)
        E_mag = (np.e**(1j*k*r)/r)*k**2  # replace with distribution of k (lambda_exc)?
        return (E_vec, E_mag, n_vec)


    def getEfield(self, theta, phi, r, curved_coords=True) -> np.ndarray:
        """propagate the E-field along r for an angle (theta_p, phi_p) in
           the pupil coordinates (?).
           Curved coords evaluates field perpendicular to ray and tangential to
           curved surface of pupil
        """
        # E = (e^ikr/r)k^2(n x p) x n
        # n = [sin(theta_p)cos(phi) i, sin(theta_p)sin(phi_p) j, cos(theta_p) k]
        n_vec = [ np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), \
            np.cos(theta)]
        n_x_p = np.cross(n_vec,self.p_vec)
        k = 2*np.pi/self.lda_exc 
 
        E_vec = np.cross(n_x_p, n_vec)
        E_mag = (np.e**(1j*k*r)/r)*k**2  # replace with distribution of k (lambda_exc)

        if curved_coords:
            # phi = 0 we preserve x and y
            E_vec = self._rotate_efield(E_vec, theta, phi)
            # should E_z really be zero? Non TEM modes in air? No
            if E_vec[2]**2 > 1e-3*(E_vec[0]**2 + E_vec[1]**2 + E_vec[2]**2):
                print("Ez =", E_vec[2], "Ex =", E_vec[0], "E_y =", E_vec[1])
                raise Exception("E_z is not zero in ray's frame!")

        return (E_vec, E_mag, n_vec)
    
    def _rotate_efield(self, E_vec, theta_polar, phi):

        E_x_tf = E_vec[0]*np.cos(phi)*np.cos(theta_polar) + \
            E_vec[1]*np.sin(phi)*np.cos(theta_polar)\
            - E_vec[2]*np.sin(theta_polar)
        E_y_tf = -E_vec[0]*np.sin(phi) + E_vec[1]*np.cos(phi)
        # E_z_tf should equal 0
        E_z_tf = E_vec[0]*np.sin(theta_polar)*np.cos(phi)\
            + E_vec[1]*np.sin(theta_polar)*np.sin(phi) + E_vec[2]*np.cos(theta_polar)
        E_rot = [E_x_tf, E_y_tf, E_z_tf]

        return E_rot
        """
        gamma = np.arcsin(np.sin(theta_polar)*np.sin(phi))
        theta = np.arcsin(np.sin(theta_polar)*np.cos(phi))
        E_x_tf = E_vec[0]*np.cos(phi)*np.cos(theta) +\
            E_vec[1]*np.sin(phi)*np.cos(theta) -\
            E_vec[2]*np.sin(theta)
        E_y_tf = E_vec[0]*(np.cos(phi)*np.sin(theta)*np.sin(gamma) - np.sin(phi)*np.cos(gamma)) +\
            E_vec[1]*(np.sin(phi)*np.sin(theta)*np.sin(gamma) + np.cos(phi)*np.cos(gamma)) +\
            E_vec[2]*(np.cos(theta)*np.sin(gamma))
        # E_z_tf should equal 0
        E_z_tf = E_vec[0]*(np.cos(phi)*np.sin(theta)*np.cos(gamma) + np.sin(phi)*np.sin(gamma)) +\
            E_vec[1]*(np.sin(phi)*np.sin(theta)*np.cos(gamma) - np.cos(phi)*np.sin(gamma)) +\
            E_vec[2]*np.cos(theta)*np.cos(gamma) 
        E_rot = [E_x_tf, E_y_tf, E_z_tf]

        return E_rot
        """

    def new_ray(self, theta, phi, z):
        """calculate new E-field based on position in pupil at z defined by (theta, phi) and dipole position (dx dy dz)"""
        if not self.ray.exists():
            # use getEfield to calculate the E field based on a propagation vector
            # and the dipole distribution, i.e. this is used for the calculation
            # between the dipole and entrance pupil, then standard ray tracing is
            # used after that. In reality since calculating the actual entrance
            # pupil might be hard, I might just completely overfill the pupil
            # and lose rays along the way (not very efficient but hey)

            # z = working distance if first element in ray tracing is objective
            E_vec, E_mag, k_vec = self.getEfield(theta, phi, z)

            ## now get polarisation: E vector relative to k, only azimuthal
            # convert E and k to useful things for the ray
            # i.e. do Rz and Ry coord transforms, if they're correct, Ez should be
            # zero in the new coords, make a func for this?

            E_vec = self._rotate_efield(E_vec, theta, phi)
            if E_vec[2]**2 > 1e-4*(E_vec[0]**2 + E_vec[1]**2):
                print("Ez = ", E_vec[2])
                raise Exception("E_z is not zero in ray's frame!")
            magnitude = (E_vec[0]**2 + E_vec[1]**2 + E_vec[2]**2)**0.5
            polarisation = \
                E_vec/magnitude

            ray = Ray(self.lda_exc, polarisation, k_vec, magnitude, E_mag)

    def generate_pupil_rays(self):
        pass

    def generate_pupil_field(self, NA, r=1, return_coords=False,\
        phi_points=100, sin_theta_points=50):
        """
        For the polar color heatmaps we need the separate angle ranges i.e. something like
        [0, pi] and [0, pi, 2pi]  (length n and m each), while coords are like
        [0, 0, 0, pi, pi, pi] and [0, pi, 2pi, 0, pi, 2pi] (aka both length = nxm)
        set return_coords=True to return coords instead of the separate angle rangess
        """
        max_sin_theta = NA  # assume n = 1
        pupil_sin_theta_range = np.linspace(0,max_sin_theta,sin_theta_points)
        pupil_theta_range = np.arcsin(pupil_sin_theta_range)
        pupil_phi_range = np.linspace(0,2*np.pi,phi_points)

        pupil_vals_x = np.zeros([len(pupil_theta_range), len(pupil_phi_range)])
        pupil_vals_y = np.zeros([len(pupil_theta_range), len(pupil_phi_range)])
        pupil_vals_mag = np.zeros([len(pupil_theta_range), len(pupil_phi_range)],\
            dtype=np.complex_)  # the mag prefactor has a phase term

        phi_list = np.zeros([len(pupil_theta_range) * len(pupil_phi_range)])
        sin_theta_list = np.zeros([len(pupil_theta_range) * len(pupil_phi_range)])

        # evaluate the field across the pupil
        i = 0
        for t_i, theta in enumerate(pupil_theta_range):
            for p_i, phi in enumerate(pupil_phi_range):
                e_vec, e_mag, k_vec = self.getEfield(theta, phi, r)
                # print(e_vec)
                # print('######')
                pupil_vals_x[t_i, p_i] = e_vec[0]
                pupil_vals_y[t_i, p_i] = e_vec[1]
                pupil_vals_mag[t_i, p_i] = e_mag

                phi_list[i] = pupil_phi_range[p_i]
                sin_theta_list[i] = pupil_sin_theta_range[t_i]
                i += 1

        #grid of values, interpolated a bit more?

        grid_sin_theta, grid_phi = np.meshgrid(pupil_sin_theta_range, pupil_phi_range)
        
        if return_coords:
            points = (sin_theta_list, phi_list)

        vals_efield_x = pupil_vals_x.flatten()
        vals_efield_y = pupil_vals_y.flatten()

        return (pupil_phi_range,pupil_sin_theta_range), pupil_vals_x, pupil_vals_y
    

class ComplexDipole(Dipole):
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
        else:
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

    def depolarise(self, direction=None):
        """only considers rotation in one dimension (for now? okay who am I kidding)
        Arguments:
            direction <1 or -1> -- explicit choice of diffusion direction in theta
        """
        # use Perin equation (Lakowicz, principles of fluoroscopy - chapter 10)
        D_corr = (6 * self.correlation)**(-1)
        mean_squ_rotation = 2*D_corr*self.lifetime

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
        else:
            if direction != -1 and direction != 1:
                raise(ValueError("Direction of diffusion must be -1 or 1"))
        self.theta_dipole_coords = \
            (direction==1)*theta_t_positive + (direction==-1)*theta_t_negative
        




