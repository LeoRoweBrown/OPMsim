from copy import deepcopy
from typing import no_type_check

import numpy as np
import ray
from tools import printif
from scipy.interpolate import griddata

class Dipole:
    def __init__(
        self, phi_d, alpha_d, lda_em=None, lda_exc=None) -> None:
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

        """
        if lda_exc is None:
            lda_exc = 500e-9  # should be a distribution at some point

        # self.theta = (theta + np.pi/2) % (2*np.pi)  # 'theta' for rays (from x axis)
        self.alpha_d = alpha_d
        # self.theta_dipole_coords = theta  # 'theta' definition for dipole (from y axis)
        self.phi_d = phi_d
        self.lda_em = lda_em
        self.lda_exc = lda_exc
        self.density = 1  # scales quantity/density/fractional quanitity?
        # expression of p has polar angle measured from z not x
        alpha_90deg = (alpha_d + np.pi/2) % (2*np.pi)

        # get dipole angle
        # angles given with theta measured from x not z so cos(theta) <-> sin(theta)
        self.p_vec = [ np.cos(alpha_d)*np.cos(phi_d),\
            np.cos(alpha_d)*np.sin(phi_d), \
            np.sin(alpha_d)]
    
    def getEfield_z(self, phi, theta, z=1, dx=0, dy=0, dz=0):
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
        n_vec = [z*np.tan(theta)*np.cos(phi), z*np.tan(theta)*np.sin(phi), z]
        n_x_p = np.cross(n_vec,self.p_vec)
        k = 2*np.pi/self.lda_exc

        r = (n_vec[0]**2 + n_vec[1]**2 + n_vec[2]**2)**0.5
        n_vec /= r  # normalise
 
        E_vec = np.cross(n_x_p, n_vec)
        E_mag_real = 1/r  # just do 1/r for testing
        E_mag = (np.e**(1j*k*r)/r)*k**2  # replace with distribution of k (lambda_exc)?
        E_vec = E_vec.astype(complex)   # hope this doesnt slow things too much

        return (E_vec, E_mag, n_vec)

    def getEfield(self, phi, theta, r, curved_coords=True,
        xy_basis=True) -> np.ndarray:
        """propagate the E-field along r for an angle (theta, phi) in
           the pupil coordinates (?).
           Curved coords evaluates field perpendicular to ray and tangential to
           curved surface of pupil - NOTE: x and y basis is also rotated, so
           when evalulating call with xy_basis=True if you want xy polarisation
           to be in the x-y basis of the system
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
        n_vec = [ np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), \
            np.cos(theta)]
        n_x_p = np.cross(n_vec,self.p_vec)
        k = 2*np.pi/self.lda_exc 
 
        E_vec = np.cross(n_x_p, n_vec)
        E_mag = (np.e**(1j*k*r)/r)*k**2  # replace with distribution of k (lambda_exc)

        E_vec = E_vec.astype(complex)   # hope this doesnt slow things too much

        # in ray tracing not needed because we do the refraction
        # MAYBE remove all this at some point to avoid confusion
        if curved_coords:  # rotate coords so E_z in this frame = 0
            # phi = 0 we preserve x and y
            E_vec = self._rotate_efield(E_vec, phi,  theta)
            # should E_z really be zero? Non TEM modes in air? No
            if E_vec[2]**2 > 1e-3*(E_vec[0]**2 + E_vec[1]**2 + E_vec[2]**2):
                print("Ez =", E_vec[2], "Ex =", E_vec[0], "E_y =", E_vec[1])
                raise Exception("E_z is not zero in ray's frame!")

            # now convert x and y rotated basis back to lab basis for meaningful
            # polarisation

            # xy field (polarisation) transformation to recover original x-y basis
            if xy_basis:
                E_vec_x = E_vec[0]*np.cos(phi) - E_vec[1]*np.sin(phi)
                E_vec_y = E_vec[0]*np.sin(phi) + E_vec[1]*np.cos(phi)

                E_vec[0] = E_vec_x
                E_vec[1] = E_vec_y

        return (E_vec, E_mag, n_vec)

    def getEfield_bfp(self, phi, theta, WD, xy_basis=True,
        rescale_energy=True, dtheta=0) -> np.ndarray:
        # L(-theta)R(phi)E_vec
        n_vec = [ np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), \
            np.cos(theta)]
        n_x_p = np.cross(n_vec,self.p_vec)
        k = 2*np.pi/self.lda_exc 
 
        E_vec = np.cross(n_x_p, n_vec)
        E_mag = (np.e**(1j*k*WD)/WD)*k**2  # replace with distribution of k (lambda_exc)

        E_vec = E_vec.astype(complex)   # hope this doesnt slow things too much

        # coord transform
        E_vec = self._rotate_efield(E_vec, phi, 0)  # only do R(phi)

        # lens refraction in meridonal plane
        E_vec_x = E_vec[0]*np.cos(-theta) + E_vec[2]*np.sin(-theta)
        E_vec_y = E_vec[1]
        E_vec_z = -E_vec[0]*np.sin(-theta) + E_vec[2]*np.cos(-theta)

        E_vec[0] = E_vec_x
        E_vec[1] = E_vec_y
        E_vec[2] = E_vec_z

        if xy_basis:  # leave meridonal basis, go back to lab coords
            E_vec_x = E_vec[0]*np.cos(phi) - E_vec[1]*np.sin(phi)
            E_vec_y = E_vec[0]*np.sin(phi) + E_vec[1]*np.cos(phi)

        E_vec[0] = E_vec_x
        E_vec[1] = E_vec_y

        if rescale_energy:
            if np.cos(theta)<1e-4:
                E_vec[0] = 0
                E_vec[1] = 0
                E_vec[2] = 0
                Warning("At edge of pupil, setting |E|=0")
            E_vec = E_vec/(np.cos(theta))**0.5
            

        return (E_vec, E_mag, n_vec)

    
    def _rotate_efield(self, E_vec, phi, theta_polar):
        """ changes coordinate system according to k vector so that Ez = 0 """
        E_x_tf = E_vec[0]*np.cos(phi)*np.cos(theta_polar) + \
            E_vec[1]*np.sin(phi)*np.cos(theta_polar)\
            - E_vec[2]*np.sin(theta_polar)
        E_y_tf = -E_vec[0]*np.sin(phi) + E_vec[1]*np.cos(phi)
        # E_z_tf should equal 0
        E_z_tf = E_vec[0]*np.sin(theta_polar)*np.cos(phi)\
            + E_vec[1]*np.sin(theta_polar)*np.sin(phi) + E_vec[2]*np.cos(theta_polar)
        E_rot = [E_x_tf, E_y_tf, E_z_tf]

        return E_rot


    def new_ray(self, theta, phi, r, include_prefactor=False):
        """
        calculate new E-field based on position in pupil defined by (theta, phi) 
        and radius of curved pupil (which gives the ray height)
        """
        # use getEfield to calculate the E field based on a propagation vector
        # and the dipole distribution, i.e. this is used for the calculation
        # between the dipole and entrance pupil, then standard ray tracing is
        # used after that.

        # xy_basis=False means we return in meridonal coords, only makes a difference
        # when curved_coords is used though
        E_vec, E_pre, k_vec = self.getEfield(phi, theta, r, curved_coords=False,\
            xy_basis=False)

        ## now get polarisation: E vector relative to k, only azimuthal
        # convert E and k to useful things for the ray
        # i.e. do Rz and Ry coord transforms, if they're correct, Ez should be
        # zero in the new coords, make a func for this?
        # get field in frame of ray direction, Ez = 0
        E_vec_check = self._rotate_efield(E_vec, phi, theta)
        if E_vec_check[2]**2 > 1e-4*(E_vec_check[0]**2 + E_vec_check[1]**2):
            print("Ez = ", E_vec_check[2])
            raise Exception("E_z is not zero in ray's frame!")

        # complex phase factor (often ignored), may use to parameterise finite
        # spectral, lifetime and phase differences between dipoles
        if include_prefactor:  
            E_vec *= E_pre
        
        # I_vec = E_vec * np.conjugate(E_vec)
        # magnitude = (E_vec[0]**2 + E_vec[1]**2 + E_vec[2]**2)**0.5

        # generate ray in a meridonal plane i.e. polarisation in basis aligned with phi
        # do we need k_vec input? can work out with phi, theta
        rho = r*np.sin(theta)  # ray height
        ray_ = ray.PolarRay(self.lda_exc, phi, theta, rho, E_vec, k_vec, E_pre)
        return ray_

    def generate_pupil_rays(self, NA, f, phi_points=100, theta_points=50):
        ## IS THE MISTAKE HERE? ##
        """
        Generate rays over varying theta, phi with max theta defined by NA
        Calculate E-field and use theta, phi to define k-vector and generate
        ray objects for each (theta, phi)/each point on entrance pupil(?)

        Rays are traced later on in method derived from obSTORM paper, Kim et al.
        """
        self.phi_points = phi_points
        self.theta_points = theta_points
        max_sin_theta = NA  # assume n = 1
        pupil_theta_range = np.linspace(0,np.arcsin(NA),theta_points)
        # remove this later ? or keep this kind of sampling v v
        pupil_sin_theta_range = np.linspace(0,max_sin_theta,theta_points)
        pupil_theta_range = np.arcsin(pupil_sin_theta_range)
        # pupil_phi_range = np.linspace(0,2*np.pi,phi_points,endpoint=False)
        pupil_phi_range = np.linspace(0,2*np.pi,phi_points,endpoint=True)
        self.ray_list = [None]*phi_points*theta_points

        n = 0
        for t_i, theta in enumerate(pupil_theta_range):
            for p_i, phi in enumerate(pupil_phi_range):
                self.ray_list[n] = self.new_ray(theta, phi, f, include_prefactor=False)
                n += 1

    def generate_pupil_rays_input(self, f, phi_points, theta_points, areas=None, areas_alt=None):
        """
        N_phi is array of length N_ring with the number of azimuthal rays for a ring with unique a polar angle 
        """
        num_theta_points = len(theta_points)
        num_phi_points = len(phi_points)
        if num_theta_points != num_phi_points:
            raise Exception("Array length mismatch") 
        self.ray_list = [None]*num_phi_points
        # print("ray list", len(self.ray_list))
        # print("theta/phi points", num_phi_points)

        n = 0
        for i in range(num_phi_points):
            self.ray_list[i] = self.new_ray(theta_points[i], phi_points[i], f, include_prefactor=False)

        self.phi_points = num_theta_points
        self.theta_points = num_phi_points
        self.areas = areas
        self.areas_alt = areas_alt
        
    def generate_pupil_field(self, NA, r=1, pupil='curved', return_coords=False,\
        rescale_energy=False, phi_points=100, sin_theta_points=50,\
        include_phase_factor=True):
        """
        return_coords (bool):
            For the polar color heatmaps we need the separate angle ranges i.e. something like
            [0, pi] and [0, pi, 2pi]  (length n and m each), while coords are like
            [0, 0, 0, pi, pi, pi] and [0, pi, 2pi, 0, pi, 2pi] (aka both length = nxm)
            set return_coords=True to return coords instead of the separate angle ranges

            !! return complex could be very wrong, adding E-fields without considering
            imag component !!
        """
        max_sin_theta = NA  # assume n = 1
        # sine theta sampling
        # pupil_sin_theta_range = np.linspace(0,max_sin_theta,sin_theta_points)
        # pupil_theta_range = np.arcsin(pupil_sin_theta_range)

        # test - same method as using rays REMOVE LATER
        pupil_theta_range = np.linspace(0,np.arcsin(NA),sin_theta_points)
        pupil_sin_theta_range = np.sin(pupil_theta_range)

        pupil_phi_range = np.linspace(0,2*np.pi,phi_points)

        pupil_vals_x = np.zeros([len(pupil_theta_range), len(pupil_phi_range)],\
            dtype=np.complex_)
        pupil_vals_y = np.zeros([len(pupil_theta_range), len(pupil_phi_range)],\
            dtype=np.complex_)
        pupil_vals_mag = np.zeros([len(pupil_theta_range), len(pupil_phi_range)],\
            dtype=np.complex_)  # the mag prefactor has a phase term

        phi_list = np.zeros([len(pupil_theta_range) * len(pupil_phi_range)])
        sin_theta_list = np.zeros([len(pupil_theta_range) * len(pupil_phi_range)])
        theta_list = np.zeros([len(pupil_theta_range) * len(pupil_phi_range)])

        # evaluate the field across the pupil
        i = 0
        for t_i, theta in enumerate(pupil_theta_range):
            for p_i, phi in enumerate(pupil_phi_range):
                e_vec = np.array([])
                e_mag = np.array([])
                k_vec = np.array([])
                if pupil == 'flat': # evaluate over flat surface, used for testing
                    e_vec, e_mag, k_vec = self.getEfield_z(phi, theta, r,
                        rescale_energy=rescale_energy)
                elif pupil == 'curved':
                    e_vec, e_mag, k_vec = self.getEfield(phi, theta, r)
                elif pupil == 'bfp':  # back focal plane, used for testing, refracts
                    e_vec, e_mag, k_vec = self.getEfield_bfp(phi, theta, WD=r,
                        rescale_energy=rescale_energy)
                elif pupil == 'debug_curved_tf':
                    e_vec, e_mag, k_vec = self.getEfield(phi, theta, r, curved_coords=False)
                # print(e_vec)
                # print('######')
                if not include_phase_factor:
                    e_mag = 1

                pupil_vals_x[t_i, p_i] = e_vec[0]*e_mag
                pupil_vals_y[t_i, p_i] = e_vec[1]*e_mag
                
                pupil_vals_mag[t_i, p_i] = e_mag  # unused?

                phi_list[i] = phi
                sin_theta_list[i] = np.sin(theta)
                theta_list[i] = theta
                i += 1

        #grid of values, interpolated a bit more?

        grid_p, grid_r = np.meshgrid(pupil_phi_range, pupil_sin_theta_range)
        points = (sin_theta_list, phi_list)

        # if return_coords:
        #     points = (sin_theta_list, phi_list)

        vals_efield_x = pupil_vals_x.flatten()
        vals_efield_y = pupil_vals_y.flatten()

        # interp into sine projection
        interp_intensity_x = griddata(points, vals_efield_x, (grid_r, grid_p),\
            method='cubic',fill_value=0)
        interp_intensity_y = griddata(points, vals_efield_y, (grid_r, grid_p),\
            method='cubic',fill_value=0)

        return (pupil_phi_range,pupil_sin_theta_range), interp_intensity_x, interp_intensity_y
    

class ComplexDipole(Dipole):
    """Includes beta, time correlation/lifetime, spectral fluoresence"""
    def __init__(
        self, phi_d, alpha_d, lda_em=None, lda_exc=None, beta=0, lifetime=None,
        correlation=None) -> None:
        """dipole class to simulate a dipole emitter
        Arguments:
            phi_d <float> -- azimuthal angle, rotation about z axis [rad]
                           clockwise when looking along positive z (into paper)
            alpha_d <float> -- angle of dipole from x axis (note that ray calculations
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
        self.alpha_exc = alpha_d
        alpha = alpha_d + beta

        self.alpha = alpha  # 'theta' definition for dipole (from y axis)
        # calculate new alpha after diffusion (random sample this? decay stats?) #
        self.depolarise()

        # now convert these to the system coordinates
        self.alpha = (alpha_d) % (2*np.pi)  # 'theta' for rays (from z axis)
        alpha_90deg = (alpha_d + np.pi/2) % (2*np.pi)
        
        self.phi_d = phi_d
        self.lda_em = lda_em
        self.lda_exc = lda_exc

        # get dipole angle
        # angles given with theta measured from x not z so cos(theta) <-> sin(theta)
        self.p_vec = [ np.cos(alpha_d)*np.cos(phi_d),\
            np.cos(alpha_d)*np.sin(phi_d), \
            np.sin(alpha_d)]

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
        




