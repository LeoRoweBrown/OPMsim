from matplotlib import pyplot as plt
import numpy as np
import dipole
import dipole_distribution_generator as dipole_distribution
from tools import printif
import graphics
from copy import deepcopy

class DipoleSource:
    """
    Source made of multiple dipoles
    """
    def __init__(self, name=None):
        self.name = name
        self.dipole_ensemble = []
        self.dipole_info = {}
        self.dipole_info['lda_exc'] = []
        self.dipole_info['phi_d'] = []
        self.dipole_info['alpha_d'] = []
        self.excitation_polarisation = None
        self.ray_count = None

        # self.generate_dipoles(dipole_count)

    def add_dipoles(self, phi_d, alpha_d, 
        dipole_count=1, wavelength=500e-9, show_prints=True):
        """ 
        Add identical dipoles to the source, doesn't support beta, slow tumbling etc.
        Angles in degrees, phi_d and theta_d are the dipole angles
        (in the dipole coords). Alpha_d is rotation about y axis measured from x,
        phi_d is rotation about z axis with positive phi_d rotating the dipole from
        aligned with +x to +y
        """
        # for previous calls of add_dipoles
        len_start = len(self.dipole_info['phi_d'])
        self.dipole_info['lda_exc'].extend(np.ones(dipole_count)*wavelength)
        self.dipole_info['phi_d'].extend(np.ones(dipole_count)*phi_d)
        self.dipole_info['alpha_d'].extend(np.ones(dipole_count)*alpha_d)
        
        printif("Generating %d dipoles" % dipole_count, show_prints)   
        for n in range(dipole_count):
            
            p = self.dipole_info['phi_d'][len_start+n]
            a = self.dipole_info['alpha_d'][len_start+n]
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

        # areas labelled j=1 to n+2 where n is number of collars, 
        # collars are j=2 to n+1, caps are j=1 and n+2
        A_j = [2*np.pi*(np.cos(theta_c + (j-2)*delta_fitting) -\
            np.cos(theta_c + (j-1)*delta_fitting))\
            for j in range(2,(n_collars_fitting+1)+1)]

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
        # get closest match to NA
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

            # determine arrangement of points in each ring - to space as much as possible
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

        manual_area_sum = np.sum(areas_alt_k)

        self.NA = NA
        self.ray_area = areas_usingcaps

        costheta = (1-NA**2)**0.5
        expected_area =  2*np.pi*(1-costheta)

        print("cap method area sum", np.sum(areas_usingcaps))
        print("expected area sum", expected_area)

        self.ray_area_manual = manual_area_sum # np.sum(areas_alt_k[0:last_idx])

        for n in range(len(self.dipole_ensemble)):  # loop over dipoles
            dipole = self.dipole_ensemble[n]
            dipole.generate_rays(f, phi_k, theta_k, areas_usingcaps)
            if n==0:
                self.rays = deepcopy(dipole.rays)
            else:
                self.rays.I_vec += dipole.rays.I_vec  # add the rays 

        self.ray_count = len(phi_k)

        # I_k_dot = self.rays.I_vec.squeeze().dot(self.rays.k_vec.squeeze().T)
        # print("Dot product of I_vec and k_vec", I_k_dot)

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
        fig = plt.figure(figsize=[8,8])
        ax = fig.add_subplot(projection='polar')
        c = ax.scatter(self.rays.phi, np.sin(self.rays.theta), s=1)
        ax.set_ylim([0,1])  # show whole NA=1 pupil
        ax.set_title("Simulated ray distribution in pupil (sine projection)")
        plt.show()

    def view_pupil(self):

        x = self.rays.rho*np.cos(self.rays.phi)
        y = self.rays.rho*np.sin(self.rays.phi)

        if all(self.rays.rho == 0):
            x = np.sin(self.rays.theta)*np.cos(self.rays.phi)
            y = np.sin(self.rays.theta)*np.sin(self.rays.phi) 

        scatter = plt.figure()
        plt.scatter(self.rays.phi, self.rays.I_vec.squeeze()[:,0])
        plt.show()

        pupil = graphics.PupilPlotObject(x, y, self.rays.I_vec.squeeze()[:,0],  self.rays.I_vec.squeeze()[:,1])
        pplot = pupil.plot()

        print("minimum of I at start", np.min(self.rays.I_vec))
        test_rays = deepcopy(self.rays)
        test_rays.meridional_transform()
        print("minimum of I at start after merid", np.min(test_rays.I_vec))
        pupil_merid = graphics.PupilPlotObject(x, y, test_rays.I_vec.squeeze()[:,0],  test_rays.I_vec.squeeze()[:,1])
        pplot_merid = pupil_merid.plot("merid")
        test_rays.meridional_transform(True)
        print("minimum of I at start after merid and reverse merid", np.min(test_rays.I_vec))

