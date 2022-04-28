from copy import deepcopy
from matplotlib.pyplot import grid
import numpy as np
from numpy.ma.core import masked_greater
import optical_matrices
import scipy.interpolate as interp
import anisotropy
from coord_tf import get_final_polar_coords

def combine_detectors(detectors):
    """combine field from separate detectors - necessary because separate
    detector used for each dipole in parallelisation"""
    n_detectors = len(detectors)
    # arbitrarily choose first detector as the one to return
    detector = detectors[0]
    for n in range(n_detectors):
        detector.current_ifield_x += detectors[n].current_ifield_x
        detector.current_ifield_y += detectors[n].current_ifield_y
    return detector

class FlatDetector():
    def __init__(self, max_radius=None, resolution=[75,25]) -> None:
        """
        max_theta=None means auto-scale size to ray with largest theta
        resolution means number of sample points for azumithal and radial
        """
        self.max_radius = max_radius
        self.interpolated = False
    
    def view_field(self, rays):
        raise NotImplementedError()
        theta_range = 0

class Detector():
    """Like a photodetector, mapped to wavefront surface"""
    def __init__(self, curved=True, max_radius=None, resolution=None):
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
        self.is_binning_detector = False
        self.Ix_integral = None  # these are not implemented yet
        self.Iy_integral = None
        self.I_total_integral = None



    # def get_final_polar_coords(self, r, phi):
    #     """polar plot does not deal with negative radii, r=theta or rho"""
    #     if r < 0:
    #         # TEMPORARY COMMENT OUT
    #        # phi = (phi + np.pi) % (2*np.pi)
    #         # phi = (phi + np.pi)
    #         phi = phi + np.pi
    #         mask = (phi > 2*np.pi)*1
    #         phi = phi - mask*2*np.pi
    #         r = -r
    #     return r, phi

    def add_field_fast(self, dipole):
        ## if using the finalise coordinates comamand in trace_systems.py
        # COORDS MUST BE TRANSFORMED FIRST IF USING THIS, otherwise use _add_field_fast

        rays = dipole.ray_list

        if self.resolution is None:
            self.resolution = [np.max(dipole.phi_points), dipole.theta_points]

        ray_Ex_re = np.array([None]*len(rays))
        ray_Ey_re = np.array([None]*len(rays))
        ray_Ex_im = np.array([None]*len(rays))
        ray_Ey_im = np.array([None]*len(rays))
        E_vec = 0j

        # self.handle_negative_theta(rays)
        if self.ray_polar_radius == []:
            self.ray_polar_radius = np.array([None]*len(rays))
            self.ray_phi = np.array([None]*len(rays))

            for n, ray in enumerate(rays):
                self.ray_polar_radius[n] = ray.polar_radius
                self.ray_phi[n] = ray.phi
                if self.ray_phi[n] > 2*np.pi:
                    Warning("Phi bigger than 2pi")

        for n, ray in enumerate(rays):
            E_vec = ray.E_vec

            ray_Ex_re[n] = np.real(E_vec[0])
            ray_Ey_re[n] = np.real(E_vec[1])
            ray_Ex_im[n] = np.imag(E_vec[0])
            ray_Ey_im[n] = np.imag(E_vec[1])

            # I don't do cosine scaling here, the view is just distorted not image/focused
        
        self.Ix_raw += \
            (ray_Ex_re*ray_Ex_re + ray_Ex_im*ray_Ex_im)*dipole.density
        self.Iy_raw += \
            (ray_Ey_re*ray_Ey_re + ray_Ey_im*ray_Ey_im)*dipole.density


    def _add_field_fast(self, dipole):
        # polar radius in spherical pupil
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

        # self.handle_negative_theta(rays)

        for n, ray in enumerate(rays):
            #ray = deepcopy(ray)  # so we don't mutate the ray and can use for anisotropy?
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
                # ray.theta, ray_phi[n] = self.get_final_polar_coords(ray, curved=self.curved)
                # if ray.theta < 0:
                #     raise Warning("negative radius in polar plot!")
                # ray_polar_radius[n] = np.sin(ray.theta)#attr=True))
            else:  # flat pupil
                E_vec = ray.E_vec
                # ray_polar_radius[n] = np.sin(ray.get_theta())
                ray.update_history()  # before changing from (-r, phi) -> (r, phi+180)
                # ray.rho, ray_phi[n]  = self.get_final_polar_coords(ray, curved=self.curved)
                # ray_polar_radius[n] = ray.rho

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

        # from matplotlib import pyplot as plt
        # fig = plt.figure(figsize=(10,8))
        # plt.plot(self.ray_phi, self.ray_polar_radius, linestyle=None, marker='o')
        # np.set_printoptions(threshold=np.inf)
        # print("len -", len(self.ray_phi))
        # print("phis -------------------")
        # print(self.ray_phi)
        # print("------------------------")
        # print(np.min(self.ray_phi), np.max(self.ray_phi))
        # plt.show()
        # plt.hist(self.ray_phi)
        # plt.show()
        from matplotlib import pyplot as plt
        # f = plt.figure()
        # plt.hist(self.Ix_raw)
        # plt.show()

        np.set_printoptions(threshold=1000)


    def interpolate_intensity(self, wraparound=True):

        if self.max_polar_radius is None:
            self.max_polar_radius = np.max(self.ray_polar_radius)
        pupil_radius_range =  np.linspace(0, self.max_polar_radius, self.resolution[1])
        pupil_phi_range = np.linspace(0,2*np.pi,self.resolution[0])

        self.ray_r_list = pupil_radius_range
        self.ray_phi_list = pupil_phi_range

        points = (self.ray_phi, self.ray_polar_radius)
        # then interpolate for the resolution
        grid_p, grid_r = np.meshgrid(pupil_phi_range, pupil_radius_range)

        """ here I attempt to do wraparound interpolation, needs work
        # occupy grid
        Ix_grid = np.zeros_like(grid_p)
        Iy_grid = np.zeros_like(grid_p)

        Ix_grid = np.reshape(self.Ix_raw, np.shape(Ix_grid))
        Iy_grid = np.reshape(self.Ix_raw, np.shape(Ix_grid))

        # add end and beginning for circular interp
        wraparound_Ix = np.concatenate(Ix_grid[[-1],:], Ix_grid, Ix_grid[[0],:])
        wraparound_Iy = np.concatenate(Iy_grid[[-1],:], Iy_grid, Iy_grid[[0],:])
        wraparound_grid_p = np.concatenate(grid_p[[-1],:], grid_p, grid_p[[0],:])
        wraparound_grid_r = np.concatenate(grid_r[[-1],:], grid_r, grid_r[[0],:])

        if wraparound:
            wraparound_current_ifield_x = interp.griddata(points, wraparound_Ix,\
                (wraparound_grid_p, wraparound_grid_r), method='cubic',fill_value=0)
            wraparound_current_ifield_y = interp.griddata(points, wraparound_Iy,\
                (wraparound_grid_p, wraparound_grid_r), method='cubic',fill_value=0)
            self.current_ifield_x = wraparound_current_ifield_x[1:-1, :]
            self.current_ifield_y = wraparound_current_ifield_y[1:-1, :]
        """    
        self.current_ifield_x = interp.griddata(points, self.Ix_raw, (grid_p, grid_r),\
            method='cubic',fill_value=0)
        self.current_ifield_y = interp.griddata(points, self.Iy_raw, (grid_p, grid_r),\
            method='cubic',fill_value=0)
        self.interpolated = True


###### TODO: probably remove this method, it is very slow (interpolates for each dipole!)

    def add_field(self, dipole):
        # polar radius in spherical pupil
        rays = dipole.ray_list

        if self.resolution is None:
            self.resolution = [np.max(dipole.phi_points), dipole.theta_points]

        ray_Ex_re = np.array([None]*len(rays))
        ray_Ey_re = np.array([None]*len(rays))
        ray_Ex_im = np.array([None]*len(rays))
        ray_Ey_im = np.array([None]*len(rays))
        ray_polar_radius = np.array([None]*len(rays))
        ray_phi = np.array([None]*len(rays))

        ray_Ex_cx = np.array([None]*len(rays), dtype=np.complex_)
        ray_Ey_cx = np.array([None]*len(rays), dtype=np.complex_)
        E_vec = 0j

        # self.handle_negative_theta(rays)

        for n, ray in enumerate(rays):
            ray = deepcopy(ray)  # so we don't mutate the ray and can use for anisotropy?
            if ray.isMeridional:  # put back into non meridional basis
                ray.E_vec = np.matmul(
                    optical_matrices.meridional_transform(ray.phi, inverse=True),\
                    ray.E_vec)
            # if curved we evaluate the field over a curved surface, which needs curved coord system

            if self.curved:  # curved pupil
                E_vec = self.rotate_field_curved_surface(
                    ray.E_vec, ray.phi, ray.theta)#attr=True))
                
                ray.update_history()  # before changing from (-r, phi) -> (r, phi+180)
                ray.theta, ray_phi[n] = self.get_final_polar_coords(ray.theta, ray.phi)
                if ray.theta < 0:
                    raise Warning("negative radius in polar plot!")
                ray_polar_radius[n] = np.sin(ray.theta)#attr=True))
            else:  # flat pupil
                E_vec = ray.E_vec
                # ray_polar_radius[n] = np.sin(ray.get_theta())
                ray.update_history()  # before changing from (-r, phi) -> (r, phi+180)
                ray.rho, ray_phi[n]  = self.get_final_polar_coords(ray.rho, ray.phi)
                ray_polar_radius[n] = ray.rho

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
        
        points = (ray_phi, ray_polar_radius)

        # work out why phi ~ 0, 360 is zero
        phi_points_norm = ray_phi/(2*np.pi)
        intens = ((ray_Ex_re**2 + ray_Ex_im**2) + (ray_Ex_re**2 + ray_Ex_im**2))**0.5
        #from matplotlib import pyplot as plt
        #plt.plot(phi_points_norm, intens, linestyle='None', marker='o')
        #plt.show()


        #print(list(set(ray_phi/(2*np.pi))))
        #plt.hist(ray_phi, bins=100)
        #plt.show()

        if self.max_polar_radius is None:
            self.max_polar_radius = np.max(ray_polar_radius)
        pupil_radius_range =  np.linspace(0, self.max_polar_radius, self.resolution[1])
        # polar angle in spherical pupil
        pupil_phi_range = np.linspace(0,2*np.pi,self.resolution[0])
        

        self.ray_r_list = pupil_radius_range
        self.ray_phi_list = pupil_phi_range

        # then interpolate for the resolution
        grid_p, grid_r = np.meshgrid(pupil_phi_range, pupil_radius_range)

        data_x_re = interp.griddata(points, ray_Ex_re, (grid_p, grid_r),\
            method='cubic',fill_value=0)
        data_y_re = interp.griddata(points, ray_Ey_re, (grid_p, grid_r),\
            method='cubic',fill_value=0)
        data_x_im = interp.griddata(points, ray_Ex_im, (grid_p, grid_r),\
            method='cubic',fill_value=0)
        data_y_im = interp.griddata(points, ray_Ey_im, (grid_p, grid_r),\
            method='cubic',fill_value=0)

        # different method - use this for applying classical photoselection
        ray_Ix = np.real(ray_Ex_cx)*np.real(ray_Ex_cx) + np.imag(ray_Ex_cx)*np.imag(ray_Ex_cx)
        ray_Iy = np.real(ray_Ey_cx)*np.real(ray_Ey_cx) + np.imag(ray_Ey_cx)*np.imag(ray_Ey_cx)

        intensity_x = interp.griddata(points, ray_Ix, (grid_p, grid_r),\
            method='cubic',fill_value=0)
        intensity_y = interp.griddata(points, ray_Iy, (grid_p, grid_r),\
            method='cubic',fill_value=0)
        #
        
        if self.current_ifield == []:  # make sure dims are correct/init mats
            # fields aren't really used, dipole density (essentially dipole count) cannot really
            # be used to scale field since it implies coherent addition
            # self.current_efield_x = [data_x_re, data_x_im]
            # self.current_efield_y = [data_y_re, data_y_im]
            self.current_ifield_x = (data_x_re*data_x_re + data_x_im*data_x_im)*dipole.density
            self.current_ifield_y = (data_y_re*data_y_re + data_y_im*data_y_im)*dipole.density
            self.current_ifield = self.current_ifield_x +\
                self.current_ifield_y
            self.current_ifield_x_method2 = intensity_x
            self.current_ifield_y_method2 = intensity_y
        else:
            # fields aren't really used, dipole density (essentially dipole count) cannot really
            # be used to scale field since it implies coherent addition
            # self.current_efield_x[0] += data_x_re
            # self.current_efield_x[1] += data_x_im
            # self.current_efield_y[0] += data_y_re
            # self.current_efield_y[1] += data_y_im
            # incoherent addition
            self.current_ifield_x += (data_x_re*data_x_re + data_x_im*data_x_im)*dipole.density
            self.current_ifield_y += (data_y_re*data_y_re + data_y_im*data_y_im)*dipole.density
            self.current_ifield = self.current_ifield_x +\
                self.current_ifield_y

            self.current_ifield_x_method2 += intensity_x
            self.current_ifield_y_method2 += intensity_y

    def calculate_anisotropy(self, excitation_polarisation):
        raise NotImplementedError()
        return anisotropy.calculate_anisotropy(
            self.ray_r_list, self.ray_phi_list,
            self.current_efield_x, self.current_efield_y, excitation_polarisation)

    def rotate_field_curved_surface(self, E_vec, phi, theta):
        # rotate into meridional by phi and then in theta so k perpendicular to surface
        # E_vec = deepcopy(E_vec_in) # to be safe
        E_vec = self._rotate_efield(E_vec, phi, theta)
        Ex = np.absolute(E_vec[0])
        Ey = np.absolute(E_vec[1])
        Ez = np.absolute(E_vec[2])
        if Ez**2 > 1e-3*(Ex**2 + Ey**2 + Ez**2) and Ez > 1e-9:  # the end bit is a bit fudgy
            print("Ez =", E_vec[2], "Ex =", E_vec[0], "E_y =", E_vec[1])
            raise Exception("E_z is not zero in ray's frame!")

        # now convert x and y rotated meridional basis back to lab basis for meaningful
        # polarisation

        # xy field (polarisation) transformation to recover original x-y basis
        E_vec_x = E_vec[0]*np.cos(phi) - E_vec[1]*np.sin(phi)
        E_vec_y = E_vec[0]*np.sin(phi) + E_vec[1]*np.cos(phi)

        E_vec[0] = E_vec_x
        E_vec[1] = E_vec_y

        return E_vec

    def _rotate_efield(self, E_vec, phi, theta_polar):
        """ 
        changes coordinate system according to k vector so that Ez = 0
        both rotates into meridional and then does theta rotation
        """
        E_x_tf = E_vec[0]*np.cos(phi)*np.cos(theta_polar) + \
            E_vec[1]*np.sin(phi)*np.cos(theta_polar)\
            - E_vec[2]*np.sin(theta_polar)
        E_y_tf = -E_vec[0]*np.sin(phi) + E_vec[1]*np.cos(phi)
        # E_z_tf should equal 0
        E_z_tf = E_vec[0]*np.sin(theta_polar)*np.cos(phi)\
            + E_vec[1]*np.sin(theta_polar)*np.sin(phi) + E_vec[2]*np.cos(theta_polar)
        E_rot = [E_x_tf, E_y_tf, E_z_tf]

        return E_rot

class AnisotropyDetector():
    """idea is that this doesn't mutate rays and is placed before imaging objective
    using this to check if applying 'polariser' on curved surface at the end is the same"""
    def __init__(self, excitation_polarisation) -> None:
        self.type = 'AnisotropyDetector'
        
    def calculate_anisotropy(rays):
        rays_p = deepcopy(rays)
        raise NotImplementedError()

class RealDetectorSampling():
    """Bins rays/photons instead of assigns them to pixels, is more correct
    if there are local focussing effects/density of rays changes
    TODO: actually do this"""
    def __init__(self, max_polar_radius, resolution=[100,50]) -> None:
        self.max_polar_radius = max_polar_radius
        self.resolution = resolution
    
    def view_field(self, rays):
        pass