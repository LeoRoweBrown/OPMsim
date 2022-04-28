import numpy as np
import scipy.interpolate as interp
from scipy.spatial import Delaunay
import anisotropy
from coord_tf import get_final_polar_coords
import optical_matrices
import time

class MeshDetector():
    """Bins rays/photons instead of assigns them to pixels, is more correct
    if there are local focussing effects/density of rays changes
    TODO: actually do this"""
    def __init__(self, curved=True, max_radius=None, resolution=(30,15)):
        """
        max_theta=None means auto-scale size to ray with largest rho
        resolution means number of sample points for azumithal and radial
        """
        self.max_polar_radius = max_radius  # if flat wavefront, rho, if curved, sine(theta)
        self.resolution = resolution
        self.current_ifield = []
        self.current_efield_x = [[],[]]  # real, imag
        self.current_efield_y = [[],[]]                                                                                                
        self.current_ifield_x = []
        self.current_ifield_y = []
        self.Ix_raw = 0
        self.Iy_raw = 0
        self.curved = curved  # curved pupil surface - use sine mapping to radius
        # lists of UNIQUE values for position on pupil
        self.ray_r_list = []  
        # self.ray_theta_list = []
        self.ray_phi_list = []  # 'lists' have no duplicates
        self.curved = curved
        self.ray_polar_radius = []  # these are every point
        self.ray_phi = []
        self.interpolated = False

    def add_field(self, dipole):
        """To add each dipole's contribution to the detector, must be run for each dipole"""

        rays = dipole.ray_list

        if self.resolution is None:
            self.resolution = [np.max(dipole.phi_points), dipole.theta_points]

        ray_Ex_re = np.array([None]*len(rays))
        ray_Ey_re = np.array([None]*len(rays))
        ray_Ex_im = np.array([None]*len(rays))
        ray_Ey_im = np.array([None]*len(rays))
        ray_polar_radius = np.array([None]*len(rays))
        ray_phi = np.array([None]*len(rays))
        ray_phi_not_mod = np.array([None]*len(rays))

        ray_Ex_cx = np.array([None]*len(rays), dtype=np.complex_)
        ray_Ey_cx = np.array([None]*len(rays), dtype=np.complex_)
        E_vec = 0j


        for n, ray in enumerate(rays):
            if ray.isMeridional:  # put back into non meridional basis
                ray.E_vec = np.matmul(
                    optical_matrices.meridional_transform(ray.phi, inverse=True),\
                    ray.E_vec)
                ray.isMeridional = False
            # if curved we evaluate the field over a curved surface, which needs curved coord system

            if self.curved:  # curved pupil
                E_vec = self.rotate_field_curved_surface(
                    ray.E_vec, ray.phi, ray.theta)#attr=True))
                
                ray.update_history()  # before changing from (-r, phi) -> (r, phi+180)
                ray_phi_not_mod[n] = ray.phi
            else:  # flat pupil
                E_vec = ray.E_vec
                ray.update_history()  # before changing from (-r, phi) -> (r, phi+180)

            ray_polar_radius[n], ray_phi[n] = get_final_polar_coords(ray, curved=self.curved)

            self.ray_polar_radius = ray_polar_radius
            self.ray_phi = ray_phi

            if ray_phi[n] > 2*np.pi:
                Warning("Phi bigger than 2pi")

            ray_Ex_re[n] = np.real(E_vec[0])
            ray_Ey_re[n] = np.real(E_vec[1])
            ray_Ex_im[n] = np.imag(E_vec[0])
            ray_Ey_im[n] = np.imag(E_vec[1])

            # used in different method
            ray_Ex_cx[n] = E_vec[0]
            ray_Ey_cx[n] = E_vec[1]

            # I don't do cosine scaling here, the view is just distorted not image/focused

        # different method - use this for applying classical photoselection
        
        self.Ix_raw += \
            (np.real(ray_Ex_cx)*np.real(ray_Ex_cx) + np.imag(ray_Ex_cx)*np.imag(ray_Ex_cx))*dipole.density
        self.Iy_raw += \
            (np.real(ray_Ey_cx)*np.real(ray_Ey_cx) + np.imag(ray_Ey_cx)*np.imag(ray_Ey_cx))*dipole.density


    def generate_detector_mesh(self):
        """
        Create empty matrix to represent pupil intensity
        0th index for phi and 1th index for polar radius
        """
        if self.max_polar_radius is None:
            self.max_polar_radius = np.max(self.ray_polar_radius)
        phi_pixels = self.resolution[0]
        radial_pixels = self.resolution[1]
        
        self.current_ifield_x = np.zeros((phi_pixels, radial_pixels))
        self.current_ifield_y = np.zeros((phi_pixels, radial_pixels))

    def bin_intensity(self, interpolate=False):
        """
        Bin values from each ray (which will have been summed across the different
        dipoles) into pixels of this detector

        parallelise this? maybe not worth the overheads
        """
        bin_start = time.time()

        self.generate_detector_pixels()
        phi_pixels = self.resolution[0]
        radial_pixels = self.resolution[1]

        radial_is = np.zeros(len(self.ray_phi))
        phi_is = np.zeros(len(self.ray_phi))
        from matplotlib import pyplot as plt
        f = plt.figure()
        plt.hist(self.Ix_raw)
        plt.xlabel("pixel values before bin")
        plt.show()
        # print("LENGTH RAY_PHI", len(self.ray_phi))

        for n in range(len(self.ray_phi)):
            phi_i = int(np.round(
                self.ray_phi[n]*(phi_pixels - 1)/(2*np.pi)))
            radial_i = int(np.round(
                self.ray_polar_radius[n]*(radial_pixels - 1)/(self.max_polar_radius)))
            
            radial_is[n] = radial_i
            phi_is[n] = phi_i
            # radial_vals[n] = radial_i
            # phi_vals[n] = phi_i

            if phi_i > phi_pixels:
                raise Exception("phi point exceeds range of 0-2pi!")
            if radial_i > radial_pixels:
                Warning("Radial position of ray exceeds max radius range")

            self.current_ifield_x[phi_i, radial_i] += self.Ix_raw[n]
            self.current_ifield_y[phi_i, radial_i] += self.Iy_raw[n]

        bin_time = time.time() - bin_start
        print("Time elapsed in bin section (time.time()) %fs" % bin_time)

        f = plt.figure()
        plt.hist(self.current_ifield_x.flatten())
        plt.title("after bin")
        plt.xlabel("pixel values")
        plt.show()

        plt.figure()
        plt.hist(radial_is)
        plt.xlabel("theta")
        plt.show()
        plt.figure()
        plt.hist(phi_is)
        plt.xlabel("phi")
        plt.show()

        plt.figure()
        plt.plot(phi_is)
        plt.xlabel("phi")
        plt.show()

        fig = plt.figure(figsize=[10,4])

        #Create a polar projection
        ax1 = fig.add_subplot(131, projection='polar')
        # pc1 = ax1.pcolormesh(angles,pupil_r_range,data_x.T,\
        pupil_radius_range =  np.linspace(0, self.max_polar_radius, self.resolution[1])
        pupil_phi_range = np.linspace(0,2*np.pi,self.resolution[0])

        self.ray_r_list = pupil_radius_range
        self.ray_phi_list = pupil_phi_range

        pc1 = ax1.pcolormesh(pupil_phi_range, pupil_radius_range, self.current_ifield_x.T,\
             shading='auto')
        # pc1 = ax1.pcolormesh(self.polar_radii, self.polar_angles, self.data_x,\
        #     shading='auto')
        ax1.set_title("testing polar plot")
        fig.colorbar(pc1, ax=ax1, fraction=0.045, pad=0.18)
        fig.show()

        if interpolate:
            self.interpolate_intensity()
        else:
            self.current_ifield_x = self.current_ifield_x.T
            self.current_ifield_y = self.current_ifield_y.T
            self.interpolated = True


    def interpolate_intensity(self):
        """Interpolate structued intensity data after binning"""
        pupil_radius_range =  np.linspace(0, self.max_polar_radius, self.resolution[1])
        pupil_phi_range = np.linspace(0,2*np.pi,self.resolution[0])

        self.ray_r_list = pupil_radius_range
        self.ray_phi_list = pupil_phi_range

        interp_start = time.time()

        interp_spline_x = interp.RectBivariateSpline(pupil_phi_range, pupil_radius_range, self.current_ifield_x)
        interp_spline_y = interp.RectBivariateSpline(pupil_phi_range, pupil_radius_range, self.current_ifield_y)

        # phi_mesh, r_mesh = np.meshgrid(pupil_phi_range, pupil_radius_range)
        self.current_ifield_x = interp_spline_x(pupil_radius_range, pupil_phi_range)
        self.current_ifield_y = interp_spline_y(pupil_radius_range, pupil_phi_range)

        self.interpolated = True

        interp_time = time.time() - interp_start
        print("Time elapsed in bin section (time.time()) %fs" % interp_time)
