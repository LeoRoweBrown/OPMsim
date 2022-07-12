from matplotlib import pyplot as plt
import numpy as np
import dipole
import dipole_distribution_generator as dipole_distribution
from tools import printif

class DipoleSource:
    """
    Source made of multiple dipoles
    """
    def __init__(self):
        self.dipole_ensemble = []
        self.dipole_info = {}
        self.dipole_info['lda_exc'] = []
        self.dipole_info['phi_d'] = []
        self.dipole_info['alpha_d'] = []
        self.excitation_polarisation = None
        self.ray_count = None

        # self.generate_dipoles(dipole_count)

    def add_dipoles(self, alpha_d, phi_d, 
        dipole_count=1, wavelength=500e-9, show_prints=True):
        """ 
        Add identical dipoles to the source, doesn't support beta, slow tumbling etc.
        Angles in degrees, phi_d and theta_d are the dipole angles
        (in the dipole coords). Alpha_d is rotation about y axis measured from x,
        phi_d is rotation about z axis with positive phi_d rotating the dipole from
        aligned with +x to +y
        """
        self.dipole_info['lda_exc'].extend(np.ones(dipole_count)*wavelength)
        self.dipole_info['phi_d'].extend(np.ones(dipole_count)*phi_d)
        self.dipole_info['alpha_d'].extend(np.ones(dipole_count)*alpha_d)
        

        printif("Generating %d dipoles" % dipole_count, show_prints)   
        for n in range(dipole_count):
            
            p = self.dipole_info['phi_d'][n]
            a = self.dipole_info['alpha_d'][n]
            # print("Phi:", self.dipole_info['phi'][n])
            printif("Dipole: theta=%.1f, phi_d=%.1f" % (a*180/np.pi, p*180/np.pi),\
                show_prints)

            a_dipole = dipole.Dipole(p, a, lda_exc=wavelength)
            self.dipole_ensemble.append(a_dipole)

    def generate_dipoles(self, dipole_count, wavelength=500e-9, method='random',
        show_prints=False, plot=True):
        """ 
        Generate (default: randomly) distriubted dipoles with same wavelength
        doesn't support beta, slow tumbling etc.
        When not random, dipole_count is a target

        Uniform methods:
        uniform
        uniform_rotate_gradual
        uniform_rotate_90
        """
        
        # range is phi_d 2pi and alpha_d pi
        if method == 'random':
            self.dipole_info['lda_exc'].extend(np.ones(dipole_count)*wavelength)
            self.dipole_info['phi_d'].extend(np.random.random(dipole_count)*2*np.pi)
            self.dipole_info['alpha_d'].extend(
                dipole_distribution.uniform_mc_sampler(lambda t: np.cos(t), [0, np.pi/2], dipole_count))
                
            # dipole_info['theta'] = np.random.random(dipole_count)*np.pi/2
        else:
            print("Target dipole count:", dipole_count)
            # N_rings = int(np.round(np.sqrt(dipole_count)/2))  # slight approximation
            N_rings = int(np.round((-1 + np.sqrt(1 + 2*dipole_count*np.pi))/2))
            d_alpha = np.pi/(2*(1+N_rings))
            
            alpha_range = d_alpha/2 + np.arange(N_rings+1)*d_alpha  # equally spaced
            if len(alpha_range) < 2:
                raise Exception("Too few dipoles for uniform distribution (" + dipole_count + ")")

            # number of azimuthal points for each alpha/ring
            N_phi = np.round(np.pi*dipole_count*np.cos(alpha_range)/(2*(1+N_rings)))
            N_phi = np.asarray(N_phi, int)
            for i in range(N_rings+1):
                # print(N_phi[i], alpha_range[i])
                self.dipole_info['alpha_d'].extend(
                    [alpha_range[i]]*N_phi[i]
                )
                # maybe make it start at pi/2 every other ring so we dont bias phi=0
                phi_vals = np.linspace(0, 2*np.pi, N_phi[i], endpoint=False)
                # phi_vals = np.linspace(0, 2*np.pi, N_phi[i], endpoint=True)
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
                # print(phi_vals/np.pi)
                self.dipole_info['phi_d'].extend(
                    phi_vals)
                last_phi = phi_vals  # keep track of last phi so we can place the next point in-between
            dipole_count = len(self.dipole_info['phi_d'])
            if len(self.dipole_info['phi_d']) != len(self.dipole_info['alpha_d']):
                raise IndexError("alpha_d and phi_d arrays not same length")
            print("New dipole count:", dipole_count)

        # plot on sphere - maybe move this plotting somewhere more elegant
        x = np.cos(self.dipole_info['alpha_d'])*np.sin(self.dipole_info['phi_d'])
        y = np.cos(self.dipole_info['alpha_d'])*np.cos(self.dipole_info['phi_d'])
        z = np.sin(self.dipole_info['alpha_d'])

        ## plot to verify distribution
        if plot:
            self.plot_distribution()

        printif("Generating %d dipoles" % dipole_count, show_prints)   
        for n in range(dipole_count):
            
            p = self.dipole_info['phi_d'][n]
            a = self.dipole_info['alpha_d'][n]
            # print("Phi:", self.dipole_info['phi'][n])
            printif("Dipole: theta=%.1f, phi_d=%.1f" % (a*180/np.pi, p*180/np.pi),\
                show_prints)

            random_dipole = dipole.Dipole(p, a, lda_exc=wavelength)
            self.dipole_ensemble.append(random_dipole)

    def generate_uniform_dipoles(self):
        """Bauer method"""
        raise NotImplementedError()
        pass

    def plot_distribution(self, alphas=None):
        # plot on sphere - maybe move this plotting somewhere more elegant
        x = np.cos(self.dipole_info['alpha_d'])*np.cos(self.dipole_info['phi_d'])
        y = np.cos(self.dipole_info['alpha_d'])*np.sin(self.dipole_info['phi_d'])
        z = np.sin(self.dipole_info['alpha_d'])

        ## plot to verify distribution
        f2d = plt.figure(figsize=(14, 7))
        ax_xz = f2d.add_subplot(221)
        ax_xz.scatter(z, x, s=4, alpha=alphas)
        ax_xz.set_title("Distribution of dipole points \n on sphere (ZX)")
        ax_xz.set_xlabel("z")
        ax_xz.set_ylabel("x")
        ax_xz.set_aspect('equal')
        ax_xy = f2d.add_subplot(222)
        ax_xy.set_title("Distribution of dipole points \n on sphere (YX)")
        ax_xy.scatter(y, x, s=4, alpha=alphas)
        ax_xy.set_xlabel("y")
        ax_xy.set_ylabel("x")
        ax_xy.set_aspect('equal')
        ax_zy = f2d.add_subplot(223)
        ax_zy.set_title("Distribution of dipole points \n on sphere (YZ)")
        ax_zy.scatter(z, y, s=4, alpha=alphas)
        ax_zy.set_xlabel("z")
        ax_zy.set_ylabel("y")
        ax_zy.set_aspect('equal')
        f2d.tight_layout()
        
        plt.show()

    def generate_photoselection_x(self, dipole_count, wavelength=500e-9, show_prints=False):
        """ special case of x polarisation for excitation """
        self.excitation_polarisation = (0,0)
        self.dipole_info['lda_exc'].extend(np.ones(dipole_count)*wavelength)
        self.dipole_info['phi_d'].extend(np.random.random(dipole_count)*2*np.pi)
        # cos for cylindrical coordinates uniform coverage, cos^2 for photoselection
        # thanks to symmetry about x, we can use the uniform one and only care about alpha_d
        self.dipole_info['alpha_d'].extend(
            dipole_distribution.uniform_mc_sampler(lambda t: np.cos(t)*np.cos(t)*np.cos(t), [0, np.pi/2], dipole_count))

        # maybe give this a function?
        printif("Generating %d dipoles" % dipole_count, show_prints)   
        for n in range(dipole_count):
            
            p = self.dipole_info['phi_d'][n]
            a = self.dipole_info['alpha_d'][n]
            printif("Dipole: theta=%.1f, phi_d=%.1f" % (a*180/np.pi, p*180/np.pi),\
                show_prints)

            random_dipole = dipole.Dipole(p, a, lda_exc=wavelength)
            self.dipole_ensemble.append(random_dipole)
        
        self.plot_distribution()

        # self._mc_sampler(self, pdf, input_range, N, maxiter=10000, plot=True)

    def generate_general_photoselection(self, dipole_count, excitation_polarisation,
        wavelength=500e-9, show_prints=False):
        """
        Generate dipoles based on photoselection with arbitrary 
        excitation angle (phi_exc, alpha_exc)

        dipole_count <int>: number of dipoles in source
        excitation angle [phi_exc <float>, alpha_exc <float>]:
        """
        self.excitation_polarisation = excitation_polarisation
        self.dipole_info['lda_exc'].extend(np.ones(dipole_count)*wavelength)
        phi_d, alpha_d = \
            dipole_distribution.mc_sampler_photoselection(dipole_count,
            excitation_polarisation)
        self.dipole_info['alpha_d'].extend(alpha_d)
        self.dipole_info['phi_d'].extend(phi_d)

        # use dipole info list to generate dipoles
        printif("Generating %d dipoles" % dipole_count, show_prints)   
        for n in range(dipole_count):
            
            p = self.dipole_info['phi_d'][n]
            a = self.dipole_info['alpha_d'][n]
            printif("Dipole: theta=%.1f, phi_d=%.1f" % (a*180/np.pi, p*180/np.pi),\
                show_prints)

            random_dipole = dipole.Dipole(p, a, lda_exc=wavelength)
            self.dipole_ensemble.append(random_dipole)
        
        self.plot_distribution()
    
    def classical_photoselection(self, excitation_polarisation, plot=True):
        """Scale the intensity from dipoles based on their orientation and the
        excitation polarisation"""
        dipole_count = len(self.dipole_ensemble)
        phi_exc, alpha_exc = excitation_polarisation
        self.dipole_info['density'] = np.ones(len(self.dipole_info['phi_d']))
        self.excitation_polarisation = excitation_polarisation

        for n in range(dipole_count):
            phi_d = self.dipole_info['phi_d'][n]
            alpha_d = self.dipole_info['alpha_d'][n]
            cos_d_exc = np.cos(alpha_exc)*np.cos(phi_exc)*np.cos(alpha_d)*np.cos(phi_d) +\
                np.cos(alpha_exc)*np.sin(phi_exc)*np.cos(alpha_d)*np.sin(phi_d) +\
                np.sin(alpha_exc)*np.sin(alpha_d)
            self.dipole_ensemble[n].density = cos_d_exc**2
            self.dipole_info['density'][n] = cos_d_exc**2
        
        if plot:
            alphas = self.dipole_info['density']
            self.plot_distribution(alphas)
    
    def get_rays(self, NA, f, ray_count = (100,75)):#(75,50)):
        """
        Generate rays for each dipole, preparing for each dipole's rays to be traced later on

        """
        self.NA = NA
        self.ray_count = ray_count
        phi_points, theta_points = self.ray_count
        for n in range(len(self.dipole_ensemble)):  # loop over dipoles
            current_dipole = self.dipole_ensemble[n]
            # generate the rays
            current_dipole.generate_pupil_rays(NA, f, phi_points, theta_points)
            # not sure if current_dipole is a refernce to the object, but reassign just
            # in case it isn't? remove this and test, see speed difference etc.
            self.dipole_ensemble[n] = current_dipole

    def get_rays_uniform(self, NA, f, ray_count=5000, plot_sphere=False):
        """uses spiral method Bauer et al. 2000 https://arc.aiaa.org/doi/pdf/10.2514/2.4497"""
        # adapted: N -> 2N to get half sphere, but k_max = N (N = ray_count)
        if NA < 1e-12:
            Warning("NA must be greater than zero, setting to 1e-5")
            NA = 1e-5
        print("IN NEW UNIFORM RAYS")
        k_max = ray_count
        # N scales so that same number of rays for different NA
        N = (2*ray_count - 1)/(1-(1-NA**2)**0.5)  
        zk = np.array([1 - (2*k - 1)/N for k in range(1, k_max+1)])
        theta_k = np.arccos(zk)
        L = (N*np.pi)**0.5
        phi_k = (L*theta_k) % (2*np.pi)

        for n in range(len(self.dipole_ensemble)):  # loop over dipoles
            dipole = self.dipole_ensemble[n]
            dipole.generate_pupil_rays_input(f, phi_k, theta_k)

        self.ray_count = ray_count

        if plot_sphere:
            self.plot_ray_sphere(phi_k, theta_k)
        
    def get_rays_uniform_hypercube(self, NA, f, ray_count=5000, plot_sphere=False):
        
        x = np.linspace(-1,1,int(ray_count**(1/3)))
        y = np.linspace(-1,1,int(ray_count**(1/3)))
        z = np.linspace(-1,1,int(ray_count**(1/3)))

        xm, ym, zm = np.meshgrid(np.linspace(x, y, z))

        r_vals =  xm*xm + ym*ym + zm*zm
        r_mask = r_vals < 1
        z_mask = z >= 0

        xm = xm(r_mask and z_mask)
        ym = ym(r_mask and z_mask)
        zm = zm(r_mask and z_mask)

        # now convert to polar, reject r coordinate (project onto unit sphere)?
        # replace zeroes with very small numbers e.g. 1e-15?

    def get_rays_uniform_rings(self, NA, f, ray_count=5000,\
        method='uniform_phi_inbetween', plot_sphere=False):
        """ Get equal area elements in rings for uniform rays, also compute their area"""

        N = ray_count*2
        
        region_area = 4*np.pi/N
        theta_c = np.arccos(1- 2/N)
        delta_ideal = (4*np.pi/N)**0.5
        n_collars_ideal = (np.pi - 2*theta_c)/delta_ideal
        n_collars_fitting = np.int(np.max([1, np.round(n_collars_ideal)]))
        delta_fitting = delta_ideal * n_collars_ideal/n_collars_fitting

        # areas labelled j=1 to n+2 where n is number of collars, collars are j=2 to n+1, caps are j=1 and n+2
        A_j = [2*np.pi*(np.cos(theta_c + (j-2)*delta_fitting) - np.cos(theta_c + (j-1)*delta_fitting)) for j in range(2,(n_collars_fitting+1)+1)]

        area_cap = np.pi*theta_c*theta_c
        total_area = np.sum(A_j)+2*area_cap

        n_cells_ideal = np.array(A_j)/region_area

        aj = 0
        n_cells_fitting = np.zeros(n_collars_fitting)
        for j in range(n_collars_fitting):
            n_cells_fitting[j] = np.round(n_cells_ideal[j] + aj)
            aj = np.sum(n_cells_ideal[0:j+1] - n_cells_fitting[0:j+1])

        n_cells_fitting = np.concatenate([[1], n_cells_fitting])
        thetas = \
            [np.arccos(1 - (2/N)*np.sum(n_cells_fitting[0:j+1])) for j in range(0, n_collars_fitting+1)]

        n_cells_fitting = np.asarray(n_cells_fitting, dtype=int)

        areas = []
        for i in range(len(A_j)):
            area = A_j[i]/n_cells_fitting[i+1]
            areas += [area]*n_cells_fitting[i+1]  # i+1 because cap

        # Scale the surface to match the NA (scale down)
        thetas=np.array(thetas)
        needed_max_theta = np.arcsin(NA)
        max_theta_idx = np.min(np.where(thetas > needed_max_theta))

        theta_scaling = needed_max_theta/thetas[max_theta_idx]
        thetas *= theta_scaling
        thetas = thetas[0:max_theta_idx+1]
        N_rings = len(thetas)-1

        area_cap_scaled = np.pi*np.sin(thetas[0])*np.sin(thetas[0])

        phi_vals_on_ring = [None]*(N_rings)
        phi_k = np.array([0])
        theta_k = np.array([0])
        area_k = np.array([area_cap_scaled])
        areas_alt_k = np.array([area_cap_scaled])
        areas_usingcaps = np.array([area_cap_scaled])

        last_idx = None

        for i in range(N_rings):  # len - 1, because points are between i and i+1
            dtheta = (thetas[i+1] - thetas[i])
            theta = (thetas[i+1] + thetas[i])/2
            circumference = 2*np.pi*np.sin(theta)
            ring_area_man = circumference*dtheta
            # dtheta becomes flat shape (approximate for small dtheta)
            area_manual = ring_area_man/n_cells_fitting[i+1]
            # accounts for curvature (correct for all dtheta)
            area_cap_method = 2*np.pi*(np.cos(thetas[i])-np.cos(thetas[i+1]))/n_cells_fitting[i+1]

            phi_vals = np.linspace(0, 2*np.pi, n_cells_fitting[i+1], endpoint=False)

            if method == 'uniform_rotate_gradual':
                phi_vals = (phi_vals + (i/(N_rings+1))*np.pi) % (2*np.pi)
            elif method == 'uniform_rotate_90':
                phi_vals = (phi_vals + (i%2)*(np.pi/2)) % (2*np.pi)
            elif method == 'uniform_rotate_random':
                phi_vals = (phi_vals + (np.random.random())*(np.pi/2)) % (2*np.pi)
            elif method == 'uniform_phi_inbetween':
                if i > 0:
                    offset = (last_phi[0] + last_phi[1])/2
                    phi_vals = (phi_vals + offset) % (2*np.pi)
            last_phi = phi_vals  # keep track of last phi so we can place the next point in-between
            phi_vals_on_ring[i] = phi_vals

            phi_k = np.append(phi_k, phi_vals)
            theta_k = np.append(theta_k, [theta]*n_cells_fitting[i+1])
            area_k = np.append(area_k, [area]*n_cells_fitting[i+1])
            areas_alt_k = np.append(areas_alt_k, [area_manual]*n_cells_fitting[i+1])
            areas_usingcaps = np.append(areas_usingcaps, [area_cap_method]*n_cells_fitting[i+1])

        # print("phi_k", phi_k)

        manual_area_sum = np.sum(areas_alt_k)

        self.NA = NA
        self.ray_area = manual_area_sum

        costheta = (1-NA**2)**0.5
        expected_area =  2*np.pi*(1-costheta)

        print("manual (cap method) area sum", manual_area_sum)
        print("expected area sum", expected_area)

        self.ray_area_manual = manual_area_sum # np.sum(areas_alt_k[0:last_idx])
        

        for n in range(len(self.dipole_ensemble)):  # loop over dipoles
            dipole = self.dipole_ensemble[n]
            # TODO: revert to normal area at some point
            dipole.generate_pupil_rays_input(f, phi_k, theta_k, areas_usingcaps)

        self.ray_count = len(phi_k)
        if plot_sphere:
            self.plot_ray_sphere(phi_k, theta_k)

    
    def plot_ray_sphere(self, phi, theta, plot_histo=False):
        x = np.sin(theta)*np.cos(phi)
        y = np.sin(theta)*np.sin(phi)
        z = np.cos(theta)

        ## plot to verify distribution
        f2d = plt.figure(figsize=(14, 7))
        ax_xz = f2d.add_subplot(221)
        ax_xz.scatter(z, x, s=2)
        ax_xz.set_title("Distribution of ray points \n on sphere (ZX)")
        ax_xz.set_xlabel("z")
        ax_xz.set_ylabel("x")
        ax_xz.set_aspect('equal')
        ax_xy = f2d.add_subplot(222)
        ax_xy.set_title("Distribution of ray points \n on sphere (YX)")
        ax_xy.scatter(y, x, s=2)
        ax_xy.set_xlabel("y")
        ax_xy.set_ylabel("x")
        ax_xy.set_aspect('equal')
        ax_zy = f2d.add_subplot(223)
        ax_zy.set_title("Distribution of ray points \n on sphere (YZ)")
        ax_zy.scatter(z, y, s=2)
        ax_zy.set_xlabel("z")
        ax_zy.set_ylabel("y")
        ax_zy.set_aspect('equal')
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
        dipole = self.dipole_ensemble[0]
        theta = np.zeros(len(dipole.ray_list))
        phi = np.zeros(len(dipole.ray_list))
        for i, ray in enumerate(dipole.ray_list):
            phi[i] = ray.phi
            theta[i] = ray.theta_in
        fig = plt.figure(figsize=[8,8])
        ax = fig.add_subplot(projection='polar')
        c = ax.scatter(phi, np.sin(theta), s=1)
        ax.set_ylim([0,1])  # show whole NA=1 pupil
        ax.set_title("Simulated ray distribution in pupil (sine projection)")
        plt.show()