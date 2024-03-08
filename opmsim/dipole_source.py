from matplotlib import pyplot as plt
import numpy as np
from . import distribution_functions
from .rays import PolarRays
from .tools.misc import printif
from . import optical_matrices

class DipoleSource:
    """
    Source made of multiple dipoles

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
    def __init__(self, phi_d_array=[], alpha_d_array=[], name="source", lda_exc=None, lda_em=None):
        if lda_exc is None:
            lda_exc = 500e-9  # should be a distribution at some point
        if lda_em is None:
            lda_em = 500e-9  # should be a distribution at some point

        self.alpha_d = alpha_d_array
        self.phi_d = phi_d_array
        self.lda_em = lda_em
        self.lda_exc = lda_exc
        self.emission_scaling = np.ones((len(alpha_d_array), 1, 1, 1))  # scales quantity/emission_scaling/fractional quanitity of dipole
        self.n_dipoles = len(phi_d_array)

        self.excitation_polarisation = None
        self.rays = None

        # get dipole angle
        # angles given with theta measured from x not z so cos(theta) <-> sin(theta)

        self._get_p_vec()

    def _get_p_vec(self):
        self.p_vec = np.column_stack(
            (np.cos(self.alpha_d)*np.cos(self.phi_d),
            np.cos(self.alpha_d)*np.sin(self.phi_d),
            np.sin(self.alpha_d)))
        self.p_vec = self.p_vec.reshape((len(self.phi_d), 3))

    def get_initial_efields(self, rays):
        # use _get_efield to calculate the E field based on a propagation vector
        # and the dipole distribution
        n_vec = rays.k_vec
        p_vec = self.p_vec
        r = rays.radius

        n_dipoles = p_vec.shape[0]
        # print(p_vec.shape)
        n_rays = n_vec.shape[0]

        print("n_dipoles", n_dipoles)
        print("n_rays", n_rays)

        # prepare to broadcast, index=0: dipoles index=1:n_rays
        p_vec = np.reshape(p_vec, (n_dipoles, 1, 3))
        n_vec = np.reshape(n_vec, (1, n_rays, 3))

        n_x_p = np.cross(n_vec,p_vec)
        k = 2*np.pi/self.lda_exc 
 
        print("pvec shape", p_vec.shape)
        print("nvec shape",n_vec.shape)

        rays.E_vec = np.cross(n_x_p, n_vec)  # (N_rays, N_dipole, 3)
        rays.E_vec = rays.E_vec.reshape((n_dipoles, n_rays, 3, 1)) # E_vecs are (3x1)
        rays.E_pre  = (np.e**(1j*k*r)/r)*k**2  # replace with distribution of k (lambda_exc)
        print(rays.E_vec.shape)
        self.emission_scaling = self.emission_scaling.reshape(n_dipoles,1,1,1)
        print(self.emission_scaling.shape)
        rays.I_total_initial = np.sum(rays.E_vec * rays.E_vec*self.emission_scaling, axis=0)
        # energy per dipole
        
        rays.get_intensity(self.emission_scaling)
        rays.I_vec_initial = rays.I_vec
        rays.initial_energy = rays.I_total_norm

        print("initial energy shape", rays.initial_energy.shape)

    def add_dipoles(self, dipole_angles, show_prints=True):
        """ 
        Add dipoles to the source, doesn't support beta, slow tumbling etc.
        Angles in degrees, phi_d and theta_d are the dipole angles
        (in the dipole coords). Alpha_d is rotation about y axis measured from x,
        phi_d is rotation about z axis with positive phi_d rotating the dipole from
        aligned with +x to +y

        A bit redundant?
        """
        # for previous calls of add_dipoles
        self.phi_d = np.append(self.phi_d, dipole_angles[:][0])
        self.alpha_d = np.append(self.alpha_d, dipole_angles[:][1])
        self.emission_scaling = np.append(self.emission_scaling, np.ones(np.size(dipole_angles[:][0])))
        self.n_dipoles = len(self.phi_d)
        self._get_p_vec()
        

    def generate_dipoles(self, dipole_count, lda_ex=500e-9, lda_em=500e-9, method='uniform',
        show_prints=False, plot=False):
        """ 
        Generate (default: randomly) distriubted dipoles with same wavelength
        doesn't support beta, slow tumbling etc.
        
        TODO: make this generated in rings uniformly (not randomly)
        """
        
        # range is phi_d 2pi and alpha_d pi
        if method == 'random':
            self.phi_d = np.append(self.phi_d, np.random.random(dipole_count)*2*np.pi)
            self.alpha_d = np.append(self.alpha_d, \
                distribution_functions.uniform_mc_sampler(
                    lambda t: np.cos(t), [0, np.pi/2], dipole_count))

        elif method == 'uniform':
            phi_d, theta_d, areas = \
                distribution_functions.uniform_points_on_sphere(point_count=dipole_count, hemisphere=False)
            self.phi_d = np.append(self.phi_d, phi_d)
            self.alpha_d = np.append(self.alpha_d, np.pi/2 - theta_d)
        
        else:
            raise ValueError("Only supports 'random' or 'uniform' distriubted dipoles")

        self.n_dipoles = len(self.phi_d)
        self.emission_scaling = np.ones(len(self.phi_d))
        # print("self.n_dipoles", self.n_dipoles)
        # print("self.emission_scaling", len(self.emission_scaling))
        # self.lda_em = np.append(self.lda_em, np.ones(self.n_dipoles)*lda_em)

        ## plot to verify distribution
        if plot:
            self.plot_distribution()

        printif("Generating %d dipoles" % self.n_dipoles, show_prints)
        self._get_p_vec()

    def depolarise(self, correlation_time, fluorescence_lifetime, timepoints=100):
        # TODO: check this, very untested
        # simulate the perrin equation by rotating dipoles in a random direction
        D = correlation_time
        tau = fluorescence_lifetime
        self.p_vec = self.simulate_rotational_diffusion(
            self.p_vec, D, tau, timepoints=timepoints)

        # update angles as well # TODO CHECK THIS
        self.alpha_d = np.arcsin(self.p_vec[:,2])
        self.phi_d = np.arctan2(self.p_vec[:, 0], self.p_vec[:, 1])
        
        self.plot_distribution(self.emission_scaling)

        # fig = plt.figure()
        # ax = fig.add_subplot(projection='3d')
        # plt.plot()
        # plt.scatter([p_test[0]], [p_test[1]], [p_test[2]]) 
        # plt.scatter([p_test_rot[0]], [p_test_rot[1]], [p_test_rot[2]]) 
        # plt.show()

    def plot_distribution(self, alphas=[]):
        # plot on sphere - maybe move this plotting somewhere more elegant
        x = np.cos(self.alpha_d)*np.cos(self.phi_d)
        y = np.cos(self.alpha_d)*np.sin(self.phi_d)
        z = np.sin(self.alpha_d)

        facec_ = np.asarray(np.tile([0,0,0,0], [len(x),1]), dtype=np.float64)
        edgec_ = np.asarray(np.tile([0,0,0,1], [len(x),1]), dtype=np.float64)

        if len(alphas) == 0:
            alphas = np.ones_like(x)

        ## plot to verify distribution
        f2d = plt.figure(figsize=(14, 7))
        ax_xz = f2d.add_subplot(131)
        y_mask = y > 0 # get positive y
        facec = facec_
        edgec = edgec_
        facec[y_mask] = [0,0,0,1]
        alphas /= np.max(alphas)  # avoid a rogue alpha > 1 (yes it happens)

        edgec[:,3] *= alphas
        facec[:,3] *= alphas
 
        # facecolors = np.array(['none']*len(x))
        # ax_xz.scatter(z[y_mask], x[y_mask], s=5, alpha=alphas)
        ax_xz.scatter(x, z, s=5, facecolors=facec, edgecolors=edgec)
        # plot points with y < 0, i.e. the 'behind' points
        # ax_xz.scatter(z[np.invert(y_mask)], x[np.invert(y_mask)],\
        #     s=4, alpha=alphas, facecolors='none', edgecolors='blue')
        ax_xz.set_title("Distribution of dipole points \n on sphere (XZ)", fontsize = 18)
        ax_xz.set_xlabel("x", fontsize = 16)
        ax_xz.set_ylabel("z", fontsize = 16)
        ax_xz.set_aspect('equal')

        ax_xy = f2d.add_subplot(132)
        z_mask = z > 0 # get positive z
        facec = facec_
        edgec = edgec_
        facec[z_mask] = [0,0,0,1]
        edgec[:,3] *= alphas
        facec[:,3] *= alphas
        facecolors = np.array(['none']*len(x))
        facecolors[z_mask] = 'k'
        ax_xy.set_title("Distribution of dipole points \n on sphere (XY)", fontsize = 18)
        ax_xy.scatter(x, y, s=5, facecolors=facec, edgecolors=edgec)
        # ax_xy.scatter(y[z_mask], x[z_mask], s=5, alpha=alphas)
        # ax_xy.scatter(y[np.invert(z_mask)], x[np.invert(z_mask)],\
        #     s=4, alpha=alphas, facecolors='none', edgecolors='blue')
        ax_xy.set_xlabel("x", fontsize = 16)
        ax_xy.set_ylabel("y", fontsize = 16)
        ax_xy.set_aspect('equal')

        ax_zy = f2d.add_subplot(133)
        x_mask = x > 0
        facec = facec_
        edgec = edgec_
        facec[x_mask] = [0,0,0,1]
        edgec[:,3] *= alphas
        facec[:,3] *= alphas
        facecolors = np.array(['none']*len(x))
        facecolors[x_mask] = 'k'
        ax_zy.set_title("Distribution of dipole points \n on sphere (ZY)", fontsize = 18)
        ax_zy.scatter(z, y, s=5, facecolors=facec, edgecolors=edgec)
        # ax_zy.scatter(z[x_mask], y[x_mask], s=5, alpha=alphas)
        # ax_zy.scatter(z[np.invert(x_mask)], y[np.invert(x_mask)],\
        #     s=4, alpha=alphas, facecolors='none', edgecolors='blue')

        ax_zy.set_xlabel("z", fontsize = 16)
        ax_zy.set_ylabel("y", fontsize = 16)
        ax_zy.set_aspect('equal')
        f2d.tight_layout()

        if self.excitation_polarisation is not None:
            phi_exc, alpha_exc = self.excitation_polarisation
            print("plot exc arrow")
            x_ex = np.cos(alpha_exc)*np.cos(phi_exc)
            y_ex = np.cos(alpha_exc)*np.sin(phi_exc)
            z_ex = np.sin(alpha_exc)  # unused at moment

            w = 0.2
            if abs(x_ex) < 1e-9 and abs(z_ex) < 1e-9:
                ax_xz.plot(0,0, marker = 'o', markersize=16, markeredgecolor='k',markeredgewidth=2)
            else:
                ax_xz.arrow(0, 0, (1-w)*x_ex, (1-w)*z_ex, head_width=0.2, head_length=0.2,linewidth=2)#,length_includes_head=True)
                ax_xz.arrow(0, 0, -(1-w)*x_ex, -(1-w)*z_ex, head_width=0.2, head_length=0.2,linewidth=2)#,length_includes_head=True)

            if abs(x_ex) < 1e-9 and abs(y_ex) < 1e-9:
                ax_xy.plot(0,0, marker = 'o', markersize=16,markeredgecolor='k',markeredgewidth=2)
            else:
                ax_xy.arrow(0, 0, (1-w)*x_ex, (1-w)*y_ex, head_width=0.2, head_length=0.2,linewidth=2)#,length_includes_head=True)
                ax_xy.arrow(0, 0, -(1-w)*x_ex, -(1-w)*y_ex, head_width=0.2, head_length=0.2,linewidth=2)#,length_includes_head=True)
            if abs(z_ex) < 1e-9 and abs(y_ex) < 1e-9:
                ax_zy.plot(0,0, marker = 'o', markersize=16,markeredgecolor='k',markeredgewidth=2)
            else:
                ax_zy.arrow(0, 0, (1-w)*z_ex, (1-w)*y_ex, head_width=0.2, head_length=0.2,linewidth=2)#,length_includes_head=True)
                ax_zy.arrow(0, 0, -(1-w)*z_ex, -(1-w)*y_ex, head_width=0.2, head_length=0.2,linewidth=2)#,length_includes_head=True)
        
        plt.show()
    
    def classical_photoselection(self, excitation_polarisation, plot=True):
        """Scale the intensity from dipoles based on their orientation and the
        excitation polarisation"""
        
        if excitation_polarisation is None:
            if plot:
                alphas = np.ones_like(self.phi_d)*1/3
                self.plot_distribution(alphas)
            return
        dipole_count = len(self.phi_d)
        phi_exc, alpha_exc = excitation_polarisation
        self.emission_scaling = np.ones(len(self.phi_d))
        self.excitation_polarisation = excitation_polarisation

        phi_d = self.phi_d
        alpha_d = self.alpha_d
        ## TODO: double check this after changing x to y etc.
        cos_d_exc = np.cos(alpha_exc)*np.cos(phi_exc)*np.cos(alpha_d)*np.cos(phi_d) +\
            np.cos(alpha_exc)*np.sin(phi_exc)*np.cos(alpha_d)*np.sin(phi_d) +\
            np.sin(alpha_exc)*np.sin(alpha_d)
        self.emission_scaling = cos_d_exc**2  # normalise to 1
        
        if plot:
            alphas = self.emission_scaling
            self.plot_distribution(alphas)

    def plot_ray_sphere(self, phi, theta, plot_histo=False):
        x = np.sin(theta)*np.sin(phi)
        y = np.sin(theta)*np.cos(phi)
        z = np.cos(theta)
        zeros = np.zeros_like(x)

        ## plot to verify distribution
        f2d = plt.figure(figsize=(14, 7))
        ax_xz = f2d.add_subplot(131)
        ax_xz.scatter(z, x, s=2)
        ax_xz.set_title("Distribution of ray points \n on sphere (ZX)", fontsize = 12)
        ax_xz.set_xlabel("z", fontsize = 11)
        ax_xz.set_ylabel("x", fontsize = 11)
        ax_xz.set_aspect('equal')
        for i in range(len(z)):
            ax_xz.plot([0,z[i]],[0,x[i]], color=[0,0,0,0.15])
        ax_xy = f2d.add_subplot(132)
        ax_xy.set_title("Distribution of ray points \n on sphere (YX)", fontsize = 12)
        ax_xy.scatter(y, x, s=2)
        ax_xy.set_xlabel("y", fontsize = 11)
        ax_xy.set_ylabel("x", fontsize = 11)
        ax_xy.set_aspect('equal')
        for i in range(len(y)):
            ax_xy.plot([0,y[i]],[0,x[i]], color=[0,0,0,0.15])
        ax_zy = f2d.add_subplot(133)
        ax_zy.set_title("Distribution of ray points \n on sphere (YZ)", fontsize = 12)
        ax_zy.scatter(z, y, s=2)
        ax_zy.set_xlabel("z", fontsize = 11)
        ax_zy.set_ylabel("y", fontsize = 11)
        ax_zy.set_aspect('equal')
        for i in range(len(y)):
            ax_zy.plot([0,z[i]],[0,y[i]], color=[0,0,0,0.15])
        f2d.tight_layout()       
        plt.show()

        # also plot histos
        if plot_histo:
            sort_phi = list(set(phi))
            sort_phi.sort()
            print("max phi", np.max(phi))
            print("phi sep", sort_phi[1]-sort_phi[0])
            plt.hist(phi, bins=50)
            plt.xlabel(phi)
            plt.show()

    def display_pupil_rays(self):
        fig = plt.figure(figsize=[8,8])
        ax = fig.add_subplot(projection='polar')
        c = ax.scatter(self.rays.phi, np.sin(self.rays.theta), s=1)
        ax.set_ylim([0,1])  # show whole NA=1 pupil
        ax.set_title("Simulated ray distribution in pupil (sine projection)")
        plt.show()

        
    def get_2pi_energy(self, f, ray_count, ray_dist='uniform_phi_inbetween'):
        if ray_dist == "uniform":
            phi_k, theta_k, areas = distribution_functions.uniform_points_on_sphere(
                1, point_count=ray_count, method='uniform_phi_inbetween',hemisphere=True)
        elif ray_dist == "fibonacci":
            phi_k, theta_k, areas = distribution_functions.fibonacci_sphere_rays(
                1, ray_count)
            # areas = np.ones(len(phi_k))*1/len(phi_k)
        rays_1NA = PolarRays(phi_k, theta_k, f, areas, lda=self.lda_em)
        self.get_initial_efields(rays_1NA)
        print("n rays 2pi = ", rays_1NA.I_total_initial.shape[0])
        
        return rays_1NA.initial_energy, areas
        

    def get_rays_uniform_rings(self, NA, f, ray_count=5000,\
        method='uniform_phi_inbetween', plot_sphere=False, ray_dist="uniform"):
        """ Get equal area elements in rings for uniform rays, also compute their area"""

        if ray_dist == "uniform":
            phi_k, theta_k, areas = distribution_functions.uniform_points_on_sphere(
                NA, ray_count, method)
        elif ray_dist == "random":
            print("RANDOM MC RAY GENERATION")
            phi_k = np.random.random(ray_count)*2*np.pi
            theta_k = distribution_functions.uniform_mc_sampler(
                    lambda t: np.sin(t), [0, np.arcsin(NA)], ray_count)
            areas = np.ones(ray_count)*1/ray_count
            self.plot_ray_sphere(phi_k, theta_k)
        elif ray_dist == "fibonacci":
            phi_k, theta_k, areas = distribution_functions.fibonacci_sphere_rays(
                NA, ray_count)
            # areas = np.ones(len(phi_k))*1/len(phi_k)
            self.plot_ray_sphere(phi_k, theta_k)

        #print("phi_k", phi_k)
        #print("theta_k", theta_k)
        plt.figure()
        plt.scatter(phi_k,theta_k)
        plt.show()

        self.NA = NA
        self.ray_area = areas  # do we need dipole_source to have this attribute?

        # plt.figure()
        # plt.hist(areas)
        # plt.xlabel("Ray element areas")
        # plt.show()

        self.rays = PolarRays(phi_k, theta_k, f, areas, lda=self.lda_em)
        self.get_initial_efields(self.rays)
        # energy integrated over half sphere -- 
        # effectively the total integrated energy over 2pi (hemisphere solid angle)
        half_sphere_energy, area_2pi = self.get_2pi_energy(f, ray_count, ray_dist)

        print(self.rays.I_total_initial.shape)
        # plt.figure()
        # plt.hist(np.sum(self.rays.I_total_initial,axis=1))
        # plt.xlabel("Ray intensities")
        # plt.show()
        print("n rays = ", self.rays.I_total_initial.shape[0])
        # scaled to NA
        actual_area = np.sum(areas)
        print("NA area vs real area?", actual_area, 2*np.pi*(1-(1-NA**2)**0.5) )
        half_sphere_energy_NA = (1-(1-NA**2)**0.5)*half_sphere_energy
        half_sphere_energy_NA = actual_area*half_sphere_energy/np.sum(area_2pi)
        # descrbied how advantageous the collection is
        self.rays.emission_efficiency = self.rays.initial_energy/half_sphere_energy_NA
        self.rays.half_sphere_energy = half_sphere_energy
        self.rays.average_energy_times_NA = half_sphere_energy_NA

        print("half_sphere_energy",half_sphere_energy)
        print("initial_energy",self.rays.initial_energy)


        print("rays when NA scaled up to hemisphere",self.rays.I_total_initial.shape[0]*2*np.pi/actual_area)
        
        # self.emission_scaling = self.emission_scaling.reshape((len(phi_k), 1, 1, 1))

        if plot_sphere:
            self.plot_ray_sphere(phi_k, theta_k)

    def simulate_rotational_diffusion(self, p_vec, D, tau, timepoints=None):
        """
        Return new dipole orientations after depolarisation from rotational diffusion
        
        p_vec <ndarray> vector containing dipole orientations
        D <float> - correlation time in s
        tau <float> - fluorescence lifetime in s
        dt <float> time intervals (optional)
        """
        n_dipoles = np.size(p_vec, 0)
        index_list = list(range(n_dipoles))

        p_vec = p_vec.reshape(
            (p_vec.shape[0], p_vec.shape[1], 1))

        if timepoints is None:
            timepoints = 100
        dt = tau/timepoints
        # decay: exp (-t/τF)
        t = 0
        # statistical method
        while len(index_list) > 0:
            # rotate exp (-t/τF) * n_dipoles, leave the rest
            expected_n = int(np.round(np.exp(-t/tau)*n_dipoles))
            n_to_emit = len(index_list) - expected_n
            
            np.random.shuffle(index_list)  # shuffle so removal is random
                                                        # can we do this once at the beginning instead?
            if n_to_emit > len(index_list):
                n_to_emit = 1

            if n_to_emit > 0:
                index_list[0:n_to_emit-1] = []  # remove emitted entries
            
            if expected_n == 0:
                break

            n_to_rotate = len(index_list)
            # index_list = np.random.shuffle(index_list)

            ux = 2*np.random.random(n_to_rotate) - 1
            uy = 2*np.random.random(n_to_rotate) - 1
            uz = 2*np.random.random(n_to_rotate) - 1

            mag = (ux*ux + uy*uy + uz*uz)**0.5
            ux /= mag
            uy /= mag
            uz /= mag

            rot_mag = (1/D)*dt
            # print("rotate mag", rot_mag)

            rotate_mat = optical_matrices.arbitrary_rotation(\
                rot_mag, ux,uy,uz)
            rotate_mat = rotate_mat.reshape([n_to_rotate, 3,3])
            # print(np.shape(p_vec))
            # print(np.shape(p_vec[index_list,:,0]))
            # print(np.shape(rotate_mat))
            
            p_vec[index_list,:] =  rotate_mat @ p_vec[index_list,:]

            t += dt

        p_vec = p_vec.reshape(
            (p_vec.shape[0], p_vec.shape[1]))

        return p_vec