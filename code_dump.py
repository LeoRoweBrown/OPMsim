def add_field_legacy(self, rays):

        # pupil_sin_theta_range = np.linspace(0,self.max_polar_radius,self.resolution[1])
        # pupil_phi_range = np.linspace(0,2*np.pi,self.resolution[0])

        pupil_vals_x = np.zeros(len(rays), dtype=np.complex_)
        pupil_vals_y = np.zeros(len(rays), dtype=np.complex_)

        phi_list = np.zeros(len(rays), dtype=np.complex_)
        sin_theta_list = np.zeros(len(rays), dtype=np.complex_)
        theta_list = np.zeros(len(rays), dtype=np.complex_)

        # for j, ray in enumerate(rays):
        for j in range(len(rays)):
            # I ASSUME dipole.ray_list itself is mutated..?
            ray = rays[j]
            phi = ray.phi
            theta = ray.theta
            # phi = 0 we preserve x and y
            E_vec = deepcopy(ray.E_vec)
            
            E_vec = self._rotate_efield(E_vec, phi,  theta)
            # should E_z really be zero? Non TEM modes in air? No
            if E_vec[2]**2 > 1e-3*(E_vec[0]**2 + E_vec[1]**2 + E_vec[2]**2):
                print("Ez =", E_vec[2], "Ex =", E_vec[0], "E_y =", E_vec[1])
                raise Exception("E_z is not zero in ray's frame!")

            # now convert x and y rotated basis back to lab basis for meaningful
            # polarisation

            # xy field (polarisation) transformation to recover original x-y basis
            E_vec_x = E_vec[0]*np.cos(phi) - E_vec[1]*np.sin(phi)
            E_vec_y = E_vec[0]*np.sin(phi) + E_vec[1]*np.cos(phi)

            E_vec[0] = E_vec_x
            E_vec[1] = E_vec_y
            
            pupil_vals_x[j] = E_vec[0]
            pupil_vals_y[j] = E_vec[1]

            phi_list[j] = phi
            sin_theta_list[j] = np.sin(theta)
            theta_list[j] = theta

            j += 1
        
        ### more stuff here

        pupil_intensity_x = np.real(pupil_vals_x)**2 + np.imag(pupil_vals_x)**2
        pupil_intensity_y = np.real(pupil_vals_y)**2 + np.imag(pupil_vals_y)**2

        # These define interpolated stuff
        pupil_sin_theta_range = np.linspace(0, self.max_polar_radius, self.resolution[1])
        # np.linspace(0, self.max_polar_radius, self.resolution[1])
        # polar angle in spherical pupil
        pupil_phi_range = np.linspace(0, 2*np.pi, self.resolution[0])

        ## interp
        points = (phi_list, sin_theta_list)
        # pupil_sin_theta_range = np.sin(pupil_theta_range)
        grid_p, grid_r = np.meshgrid(pupil_phi_range, pupil_sin_theta_range)
        

        interp_intensity_x = interp.griddata(points, pupil_intensity_x, (grid_p, grid_r),\
            method='cubic',fill_value=0)
        interp_intensity_y = interp.griddata(points, pupil_intensity_y, (grid_p, grid_r),\
            method='cubic',fill_value=0)

        print("len interp:", len(interp_intensity_x))

        # skip interpolation FOR TESTING
        # interp_intensity_x = pupil_intensity_x
        # interp_intensity_y = pupil_intensity_y
        self.current_ifield_x = interp_intensity_x
        self.current_ifield_y = interp_intensity_y

        self.ray_r_list = pupil_sin_theta_range
        self.ray_phi_list = pupil_phi_range

    def handle_negative_theta(self, rays):
        """
        If theta is negative, it is the same as phi -> phi+pi,
        sin(theta) < 0 when theta < 0 which doesn't work on a polar plot
        cannot get this to work, maybe the phi + 180 inverse rotaion is wrong
        """
        for ray in rays:
            if ray.theta < 0:
                if ray.isMeridional:  # put back into non meridional basis
                    ray.E_vec = np.matmul(
                        optical_matrices.meridional_transform(ray.phi, inverse=True),\
                        ray.E_vec)
                    ray.isMeridional = False
                ray.E_vec = self.rotate_field_curved_surface(
                    ray.E_vec, ray.phi, ray.theta)#attr=True))

                ## rotate ray from -theta to theta
                np.matmul(optical_matrices.meridional_transform(ray.phi), ray.E_vec)
                np.matmul(optical_matrices.meridional_transform(ray.phi), ray.k_vec)
                # print("Ez =", ray.E_vec[2], "Ex =", ray.E_vec[0], "E_y =", ray.E_vec[1])
                np.matmul(optical_matrices.rotate_meridional(-2*ray.theta), ray.E_vec)
                np.matmul(optical_matrices.rotate_meridional(-2*ray.theta), ray.k_vec)
                
                if ray.E_vec[2]**2 > 1e-3*(ray.E_vec[0]**2 + ray.E_vec[1]**2 + ray.E_vec[2]**2):
                    print("Ez =", ray.E_vec[2], "Ex =", ray.E_vec[0], "E_y =", ray.E_vec[1])
                    raise Exception("E_z is not zero in ray's frame!")

                ## then rotate into phi -> phi + 180
                ray.theta = abs(ray.theta)
                ray.phi = (ray.phi + np.pi) % 2*np.pi

                np.matmul(optical_matrices.meridional_transform(\
                    ray.phi, True), ray.E_vec)
                np.matmul(optical_matrices.meridional_transform(\
                    ray.phi, True), ray.k_vec)


    def get_rays_uniform_old(self, NA, f, min_in_ring=25, ray_count=2000, method='uniform_phi_inbetween'):
        # 'uniform_phi_inbetween'
        self.ray_count = ray_count
        if type(self.ray_count) is list:
            N_rings = self.ray_count[1]
            N_phi = self.ray_count[0]
        elif type(self.ray_count) is int:
            N_rings = int(np.round((-1 + np.sqrt(1 + 2*self.ray_count*np.pi))/2))
            d_alpha = np.pi/(2*(1+N_rings)) 
            alpha_range = d_alpha/2 + np.arange(N_rings+1)*d_alpha  # equally spaced
            N_phi = np.round(np.pi*self.ray_count*np.cos(alpha_range)/(2*(1+N_rings)))
            # replace values lower than min with min
            #print(N_phi)
            N_phi = (N_phi >= min_in_ring)*N_phi + (N_phi < min_in_ring)*min_in_ring
            #print(N_phi)
        else:
            raise Exception("ray_count must be int or [int, int] for azimuth and radial points")
        if len(N_phi) < 2:
            raise Exception("ray_count or rays along phi (azimuth) not sufficient")
        N_phi = N_phi.astype(int)
        # print("N_ring", N_rings)
        # print("len phi", len(N_phi))
        max_theta = np.arcsin(NA)
        theta_vals = np.linspace(max_theta,0,len(N_phi))
        phi_vals_on_ring = [None]*(N_rings+1)
        for i in range(N_rings+1):
            # print(N_phi[i], alpha_range[i])
            # maybe make it start at pi/2 every other ring so we dont bias phi=0
            #print(N_phi[i])
            phi_vals = np.linspace(0, 2*np.pi, N_phi[i], endpoint=False)
            if method == 'uniform_rotate_gradual':
                phi_vals = (phi_vals + (i/(N_rings+1))*np.pi) % (2*np.pi)
            elif method == 'uniform_rotate_90':
                phi_vals = (phi_vals + (i%2)*(np.pi/2)) % (2*np.pi)
            elif method == 'uniform_rotate_random':
                phi_vals = (phi_vals + (np.random.random())*(np.pi/2)) % (2*np.pi)
            elif method == 'uniform_phi_inbetween':
                if i > 0:
                    # TODO: have another look at this - doesn't look right
                    offset = (last_phi[0] + last_phi[1])/2
                    phi_vals = (phi_vals + offset) % (2*np.pi)
            last_phi = phi_vals  # keep track of last phi so we can place the next point in-between
            phi_vals_on_ring[i] = phi_vals

        for n in range(len(self.dipole_ensemble)):  # loop over dipoles
            dipole = self.dipole_ensemble[n]
            dipole.generate_pupil_rays_input(NA, f, phi_vals_on_ring, theta_vals)

        self.ray_count = np.sum(N_phi)

                ## vornoi area method ##
        # TODO
        from scipy.spatial import SphericalVoronoi, geometric_slerp
        phi_v = []
        theta_v = []
        cutoff_index = 0
        print(len(n_cells_with_caps))
        print(len(thetas_with_caps))
        for i in range(len(thetas_with_caps)):
            if np.sin(thetas_with_caps[i]) > NA and i > 1 and cutoff_index == 0:
                actual_NA = (np.sin(thetas_with_caps[i-1]) + \
                    np.sin(thetas_with_caps[i]))/2
                print("NA is about", actual_NA)
                cutoff_index = len(theta_v)
            phi_vals = np.linspace(0, 2*np.pi, n_cells_with_caps[i], endpoint=False)
            if i > 0:
                if len(last_phi<2):
                    offset = 0
                else:
                    offset = (last_phi[0] + last_phi[1])/2
                phi_vals = (phi_vals + offset) % (2*np.pi)
            phi_v = np.append(phi_v, phi_vals)
            theta_v = np.append(theta_v, [thetas_with_caps[i]]*n_cells_with_caps[i])
            last_phi = phi_vals

                # convert points to cartesian
        x = np.sin(theta_v)*np.cos(phi_v)
        y = np.sin(theta_v)*np.sin(phi_v)
        z = np.cos(theta_v)
        
        points = np.array([x,y,z]).T

        radius = 1
        center = np.array([0, 0, 0])
        
        sv = SphericalVoronoi(points, radius, center)
        # vor.vertices
        # sort vertices (optional, helpful for plotting)
        sv.sort_vertices_of_regions()
        t_vals = np.linspace(0, 1, 2)
        # fig = plt.figure(figsize=(10,10))
        # ax = fig.add_subplot(111, projection='3d')
        voronoi_areas = sv.calculate_areas()