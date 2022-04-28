import numpy as np
import scipy.interpolate as interp
import scipy.stats
import anisotropy
from coord_tf import get_final_polar_coords
import optical_matrices
import time

class BinningDetector():
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
        self.ray_polar_radius = []  # these are every point
        self.ray_phi = []
        self.interpolated = False
        self.ray_areas = []
        self.is_binning_detector = True
        self.Ix_integral = None
        self.Iy_integral = None
        self.I_total_integral = None
        self.dipole_source = None

    def add_fields(self, source):
        """Loop over fields from dipole contributions"""
        for dp in source.dipole_ensemble:
            self._add_field(dp)
        self.dipole_source = source

    def _add_field(self, dipole):
        """To add each dipole's contribution to the detector, must be run for each dipole"""

        rays = dipole.ray_list

        if self.resolution is None:
            self.resolution = [np.max(dipole.phi_points), dipole.theta_points]

        ray_Ex_re = np.array([None]*len(rays))
        ray_Ey_re = np.array([None]*len(rays))
        ray_Ex_im = np.array([None]*len(rays))
        ray_Ey_im = np.array([None]*len(rays))
        ray_polar_radius = np.array([None]*len(rays), dtype=float)
        ray_phi = np.array([None]*len(rays), dtype=float)
        ray_phi_not_mod = np.array([None]*len(rays), dtype=float)

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
        self.ray_polar_radius = ray_polar_radius # .astype(float)
        self.ray_phi = ray_phi
        self.ray_areas = dipole.areas
        self.ray_areas_alt = dipole.areas_alt  # move these to dipole source

        self.Ix_raw += \
            (np.real(ray_Ex_cx)*np.real(ray_Ex_cx) + np.imag(ray_Ex_cx)*np.imag(ray_Ex_cx))*dipole.density
        self.Iy_raw += \
            (np.real(ray_Ey_cx)*np.real(ray_Ey_cx) + np.imag(ray_Ey_cx)*np.imag(ray_Ey_cx))*dipole.density

        max_for_scale = np.max(self.Ix_raw + self.Iy_raw)

        from matplotlib import pyplot as plt

        ray_phi = np.asarray(ray_phi, dtype=np.float64)

        rad = ray_polar_radius*np.cos(ray_phi)
        phi = ray_polar_radius*np.sin(ray_phi)
        print("lengths phi, rad, data", len(rad), len(phi), len(self.Ix_raw))

        """
        fig = plt.figure()

        ax = fig.add_subplot(121)
        
        pc1 = ax.tricontourf(list(rad), list(phi), self.Ix_raw, levels=256, vmin=0,vmax=max_for_scale)
        ax.set_aspect('equal')
        ax.axis('off')
        
        # new colorbar axis https://stackoverflow.com/questions/19679409/python-matplotlib-how-to-move-colorbar-without-resizing-the-heatmap 
        pc1_ax = ax.get_position()
        colorax = plt.axes([pc1_ax.x0*1.05 + pc1_ax.width * 1.05, pc1_ax.y0, 0.01, pc1_ax.height])
        fig.colorbar(pc1, cax=colorax, fraction=0.045, pad=0.18)
        ax_polar = fig.add_axes(ax.get_position(), polar=True)
        ax_polar.set_facecolor('none') # make transparent
        ax_polar.set_ylim(0, rad.max())

        ax2 = fig.add_subplot(122)
        pc2 = ax2.tricontourf(list(rad), list(phi), self.Iy_raw, levels=256, vmin=0,vmax=max_for_scale)
        ax2.set_aspect('equal')
        ax2.axis('off')

        # new colorbar axis https://stackoverflow.com/questions/19679409/python-matplotlib-how-to-move-colorbar-without-resizing-the-heatmap 
        pc2_ax = ax2.get_position()
        colorax2 = plt.axes([pc2_ax.x0*1.05 + pc2_ax.width * 1.05, pc2_ax.y0, 0.01, pc2_ax.height])
        fig.colorbar(pc2, cax=colorax2, fraction=0.045, pad=0.18)
        ax2_polar = fig.add_axes(ax.get_position(), polar=True)
        ax2_polar.set_facecolor('none') # make transparent
        ax2_polar.set_ylim(0, rad.max())

        ## graphs https://stackoverflow.com/questions/66520769/python-contour-polar-plot-from-discrete-data
        """

    def integrate_pupil(self):
        print(self.ray_areas)
        self.Ix_integral = 0
        self.Iy_integral = 0
        total_area = np.sum(self.ray_areas)
        print("total area of elements (curved)", total_area)
        total_area_flat = 0
        area_scaling = 1 
        for n in range(len(self.ray_phi)):
            if not self.curved:  # if not curved (aka odd no. lenses), adjust area elements
                theta = np.arcsin(self.ray_polar_radius[n])
                area_scaling = np.cos(theta)  # correct for the extra area due to curved surface
                total_area_flat += self.ray_areas[n]*area_scaling
            self.Ix_integral += self.ray_areas[n]*self.Ix_raw[n]*area_scaling
            self.Iy_integral += self.ray_areas[n]*self.Iy_raw[n]*area_scaling

        # area check
        print("total_area_flat", total_area_flat)
        self.Ix_integral /= total_area
        self.Iy_integral /= total_area
        self.I_total_integral = self.Ix_integral + self.Iy_integral



    def generate_detector_pixels(self):
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
        # f = plt.figure()
        # plt.hist(self.Ix_raw)
        # plt.xlabel("pixel values before bin")
        # plt.show()

        # plt.figure()
        # plt.hist(self.ray_phi, bins=100)
        # plt.xlabel("phi values before bin")
        # plt.show()
        # print("LENGTH RAY_PHI", len(self.ray_phi))

        ## numpy digitize instead?

        # +1 becuse N+1 bin boundaries for N bins
        radius_bins = np.linspace(0, self.max_polar_radius+1e-9, self.resolution[1]+1)
        # pupil_radius_range += pupil_radius_range[1] # shift so that labels are not starting from 0
        phi_bins = np.linspace(0,2*np.pi, self. resolution[0]+1)
        # + 1e-9 to include max point since binning has hard inequality on right side of bin

        pupil_phi_range = np.linspace(0,2*np.pi,self.resolution[0], endpoint=False) 
        phi_off = pupil_phi_range[1]/2
        pupil_phi_range += pupil_phi_range[1]/2  # for midpoints of 'bin'
        pupil_radius_range =  np.linspace(0, self.max_polar_radius, self.resolution[1], endpoint=False)

        digitized_phi = np.digitize(self.ray_phi, phi_bins, right=False)-1
        digitized_radius = np.digitize(self.ray_polar_radius, radius_bins, right=False)-1

        # plt.plot(hist_vals, pupil_radius_range)
        # plt.show()

        # TODO: is this right? Shouldn't this be meshgridded or something - 
        # combination of radius/phi matters - can't treat them independently.

        # scipy.stats.binned_statistics_2d <---

        # 2d matrix for binning
        # scipy.stats.binned_statistic_2d(self.ray_phi, self.ray_polar_radius, values, bins = (radius_bins, phi_bins)

        # plt.figure()
        # plt.hist(digitized_phi, bins=100)
        # plt.xlabel("phi values after numpy digitize bin ")
        # plt.show()
        # print("min dig phi", np.min(digitized_phi))
        # print("len phi dig", len(digitized_phi))
        # print("len r dig", len(digitized_radius))
        # print("size data", np.shape(self.current_ifield_x))

        # print(np.max(digitized_phi))
        # print(np.max(digitized_radius))

        for n in range(len(self.ray_phi)):

            if digitized_phi.any() >= phi_pixels:
                Warning("phi point exceeds range of 0-2pi!")
                continue
            if digitized_radius.any() >= radial_pixels:
                Warning("Radial position of ray exceeds max radius range")
                continue


            self.current_ifield_x[digitized_phi[n], digitized_radius[n]] \
                += self.Ix_raw[n]
            self.current_ifield_y[digitized_phi[n], digitized_radius[n]] \
                += self.Iy_raw[n]

        bin_time = time.time() - bin_start
        print("Time elapsed in bin section (time.time()) %fs" % bin_time)

        self.ray_r_list = pupil_radius_range
        self.ray_phi_list = pupil_phi_range

        """
        fig = plt.figure(figsize=[10,4])
        #Create a polar projection
        ax1 = fig.add_subplot(131, projection='polar')
        # pc1 = ax1.pcolormesh(angles,pupil_r_range,data_x.T,\
        # pupil_radius_range =  np.linspace(0, self.max_polar_radius, self.resolution[1])
        # pupil_phi_range = np.linspace(0,2*np.pi,self.resolution[0])

        pc1 = ax1.pcolormesh(pupil_phi_range, pupil_radius_range, self.current_ifield_x.T,\
             shading='auto')
        # pc1 = ax1.pcolormesh(self.polar_radii, self.polar_angles, self.data_x,\
        #     shading='auto')
        ax1.set_title("testing polar plot")
        # ax1.set_theta_offset(pupil_phi_range[1]/2)
        fig.colorbar(pc1, ax=ax1, fraction=0.045, pad=0.18)
        fig.show()
        """

        if interpolate:
            self.interpolate_intensity()
        else:
            self.current_ifield_x = self.current_ifield_x.T
            self.current_ifield_y = self.current_ifield_y.T
            self.interpolated = True

    def interpolate_intensity(self, wraparound=False):
        """Interpolate structued intensity data after binning"""
        ## already defined in binning, to remove
        # pupil_radius_range =  np.linspace(0, self.max_polar_radius, self.resolution[1])
        # pupil_phi_range = np.linspace(0,2*np.pi,self.resolution[0])

        # self.ray_r_list = pupil_radius_range
        # self.ray_phi_list = pupil_phi_range

        pupil_radius_range = self.ray_r_list
        pupil_phi_range = self.ray_phi_list

        points = (self.ray_phi, self.ray_r_list)
        # then interpolate for the resolution
        grid_p, grid_r = np.meshgrid(pupil_phi_range, pupil_radius_range)

        interp_start = time.time()

        interp_spline_x = interp.RectBivariateSpline(pupil_phi_range, pupil_radius_range, self.current_ifield_x)
        interp_spline_y = interp.RectBivariateSpline(pupil_phi_range, pupil_radius_range, self.current_ifield_y)

        # phi_mesh, r_mesh = np.meshgrid(pupil_phi_range, pupil_radius_range)
        self.current_ifield_x = interp_spline_x(pupil_radius_range, pupil_phi_range)
        self.current_ifield_y = interp_spline_y(pupil_radius_range, pupil_phi_range)

        # add end and beginning for circular interp
        wraparound_Ix = np.concatenate(self.Ix_raw[[-1],:], self.Ix_raw, self.Ix_raw[[0],:])
        wraparound_Iy = np.concatenate(self.Iy_raw[[-1],:], self.Iy_raw, self.Iy_raw[[0],:])
        wraparound_grid_p = np.concatenate(grid_p[[-1],:], grid_p, grid_p[[0],:])
        wraparound_grid_r = np.concatenate(grid_r[[-1],:], grid_r, grid_r[[0],:])

        if wraparound:
            wraparound_current_ifield_x = interp.griddata(points, wraparound_Ix,\
                (wraparound_grid_p, wraparound_grid_r), method='cubic',fill_value=0)
            wraparound_current_ifield_y = interp.griddata(points, wraparound_Iy,\
                (wraparound_grid_p, wraparound_grid_r), method='cubic',fill_value=0)
            self.current_ifield_x = wraparound_current_ifield_x[1:-1, :]
            self.current_ifield_y = wraparound_current_ifield_y[1:-1, :]
            
        else:
            self.current_ifield_x = interp.griddata(points, self.Ix_raw, (grid_p, grid_r),\
                method='cubic',fill_value=0)
            self.current_ifield_y = interp.griddata(points, self.Iy_raw, (grid_p, grid_r),\
                method='cubic',fill_value=0)
        self.interpolated = True


        interp_time = time.time() - interp_start
        print("Time elapsed in bin section (time.time()) %fs" % interp_time)

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