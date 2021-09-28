from matplotlib import pyplot as plt
import numpy as np
import dipole
import single_dipole
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

    def generate_dipoles(self, dipole_count, wavelength=500e-9, randomly=True, show_prints=False):
        """ 
        Generate (default: randomly) distriubted dipoles with same wavelength
        doesn't support beta, slow tumbling etc.
        When not random, dipole_count is a target
        """
        
        # range is phi_d 2pi and alpha_d pi
        if randomly:
            self.dipole_info['lda_exc'].extend(np.ones(dipole_count)*wavelength)
            self.dipole_info['phi_d'].extend(np.random.random(dipole_count)*2*np.pi)
            self.dipole_info['alpha_d'].extend(
                self._mc_sampler(lambda t: np.cos(t), [0, np.pi/2], dipole_count))
                
            # dipole_info['theta'] = np.random.random(dipole_count)*np.pi/2
        else:
            ## uniformish
            N_on_arc = int(np.ceil(dipole_count**0.5/2))
            alphas = np.linspace(0, np.pi/2, N_on_arc)
             # mean number of points on equator of given theta
            N_segments_mean = int(np.ceil(2*dipole_count**0.5)) 
            # distribution is pi/2*N_segments_mean*cos(theta)
            N_segments = np.zeros(N_on_arc, dtype=int)
            
            # scale N_segments properly so they follow cos distribution
            for n_arc in range(N_on_arc):
                N_segments[n_arc] = np.int(
                    np.ceil(np.pi/2 * N_segments_mean * np.cos(alphas[n_arc])))
                #print(N_on_arc)
                #print(N_segments[n_arc])
                self.dipole_info['alpha_d'].extend(np.tile(
                    alphas[n_arc], N_segments[n_arc]))
                
                self.dipole_info['phi_d'].extend(
                    np.linspace(0, 2*np.pi, N_segments[n_arc]))

            dipole_count = np.sum(N_segments)
            self.dipole_info['lda_exc'].extend(np.ones(dipole_count)*wavelength)

            print("New dipole count:", dipole_count)

        # plot on sphere 
        x = np.cos(self.dipole_info['alpha_d'])*np.sin(self.dipole_info['phi_d'])
        y = np.cos(self.dipole_info['alpha_d'])*np.cos(self.dipole_info['phi_d'])
        z = np.sin(self.dipole_info['alpha_d'])

        # f = plt.figure()
        # ax = f.add_subplot(111, projection='3d')
        # ax.scatter(x,y,z)
        # ax.set_box_aspect((1,1,1))
        # plt.show()

        ## plot to verify distribution
        f2d = plt.figure(figsize=(14, 7))
        ax_xz = f2d.add_subplot(211)
        ax_xz.scatter(z,x)
        ax_xz.set_title("Distribution of dipole points \n on sphere (ZX)")
        ax_xz.set_xlabel("z")
        ax_xz.set_ylabel("x")
        ax_xz.set_aspect('equal')
        ax_xy = f2d.add_subplot(221)
        ax_xy.set_title("Distribution of dipole points \n on sphere (YX)")
        ax_xy.scatter(y,x)
        ax_xy.set_xlabel("y")
        ax_xy.set_ylabel("x")
        ax_xy.set_aspect('equal')
        f2d.tight_layout()
        
        plt.show()

        printif("Generating %d dipoles" % dipole_count, show_prints)   
        for n in range(dipole_count):
            
            p = self.dipole_info['phi_d'][n]
            a = self.dipole_info['alpha_d'][n]
            # print("Phi:", self.dipole_info['phi'][n])
            printif("Dipole: theta=%.1f, phi_d=%.1f" % (a*180/np.pi, p*180/np.pi),\
                show_prints)

            random_dipole = dipole.Dipole(p, a, lda_exc=wavelength)
            self.dipole_ensemble.append(random_dipole)

    def generate_photoselection(self, dipole_count, wavelength=500e-9, show_prints=False):
        """ currently only acceping one excitation polarisation """
        # probability of excitaiton proportional to intensity, will be another MC rejection?
        raise NotImplementedError("Photoselection of dipole distribution not implemented")

    def calculate_pupil_radiation(self, NA, r=1, pupil='curved'):
        """ incoherent addition (intensity) across curved pupil for all dipoles """
        if pupil=='flat' and NA==1:
            raise Warning("Unphysical situation! NA=1 with flat pupil")
        total_intensity_x = []
        total_intensity_y = []
        angles = tuple()  # same for all dipoles
        for n in range(len(self.dipole_ensemble)):
            current_dipole = self.dipole_ensemble[n]
            angles, vals_efield_x, vals_efield_y = \
                current_dipole.generate_pupil_field(NA, r, pupil=pupil)
            intensity_x = (vals_efield_x*np.conjugate(vals_efield_x)).real
            intensity_y = (vals_efield_y*np.conjugate(vals_efield_y)).real
            if n == 0:
                total_intensity_x = intensity_x
                total_intensity_y = intensity_y
            else:
                total_intensity_x += intensity_x
                total_intensity_y += intensity_y
        return angles, total_intensity_x, total_intensity_y

    def _mc_sampler(self, pdf, input_range, N, maxiter=10000, plot=True):
        # takes normalised X~pdf(x) function and gets N points according to
        # range of x - {input range[0], input range[1]}
        print("Sampling %d points from PDF (Monte Carlo Rejection method)" % N)
        accepted_points = [0]*N
        input_range[0]
        # normalise pdf
        pdf_x_points = np.linspace(input_range[0], input_range[1], 20)
        norm = np.trapz(pdf(pdf_x_points), pdf_x_points)

        n = 0
        i = 0
        while(n < N and i < N * maxiter ):
            x_rand = input_range[0]\
                + np.random.random()*(input_range[1]-input_range[0])
            p_rand = np.random.random()
            if p_rand < pdf(x_rand)/norm:  # if under curve accept point
                accepted_points[n] = x_rand
                n += 1
            i += 1
        if plot:
            plt.hist(accepted_points, density=True)
            plot_x = np.linspace(input_range[0], input_range[1], 20)
            plt.plot(pdf_x_points, pdf(plot_x)/norm,label='PDF')
            plt.legend()
            plt.xlabel(r'$\theta$ (rad)')
            plt.ylabel('Normalised frequency')
        return accepted_points
