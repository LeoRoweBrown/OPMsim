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
        self.dipole_info['phi'] = []
        self.dipole_info['theta'] = []

        # self.generate_dipoles(dipole_count)

    def add_dipoles(self, theta, phi, dipole_count=1, wavelength=500e-9,
        dx=0, dy=0, dz=0):
        """ 
        Add identical dipoles to the source, doesn't support beta, slow tumbling etc.
        Angles in degrees, phi and theta are the dipole angles (in the dipole coords)
        Theta is rotation about y axis measured from x, phi is rotation about z axis
        with positive phi rotating the dipole from aligned with +x to +y
        """
        self.dipole_info['lda_exc'].extend(np.ones(dipole_count)*wavelength)
        self.dipole_info['phi'].extend(np.ones(dipole_count)*phi)
        self.dipole_info['theta'].extend(np.ones(dipole_count)*theta)
        self.dipole_info['displacement'].extend((dx, dy, dz))
        
        for n in range(dipole_count):
            a_dipole = dipole.Dipole(phi, theta, lda_exc=wavelength)
            self.dipole_ensemble.append(a_dipole)

    def generate_dipoles(self, dipole_count, wavelength=500e-9, show_prints=False):
        """ 
        Generate (default: randomly) distriubted dipoles with same wavelength
        doesn't support beta, slow tumbling etc.
        """

        # range is phi 2pi and theta pi
        self.dipole_info['lda_exc'].extend(np.ones(dipole_count)*wavelength)
        self.dipole_info['phi'].extend(np.random.random(dipole_count)*2*np.pi)
        self.dipole_info['theta'].extend(
            self._mc_sampler(lambda t: np.cos(t), [0, np.pi/2], dipole_count))
        # dipole_info['theta'] = np.random.random(dipole_count)*np.pi/2

        # plot on sphere 
        x = np.cos(self.dipole_info['theta'])*np.sin(self.dipole_info['phi'])
        y = np.cos(self.dipole_info['theta'])*np.cos(self.dipole_info['phi'])
        z = np.sin(self.dipole_info['theta'])

        f = plt.figure()
        ax = f.add_subplot(111, projection='3d')
        ax.scatter(x,y,z)
        ax.set_box_aspect((1,1,1))
        plt.show()

        printif("Generating %d dipoles" % dipole_count, show_prints)   
        for n in range(dipole_count):
            
            p = self.dipole_info['phi'][n]
            t = self.dipole_info['theta'][n]
            # print("Phi:", self.dipole_info['phi'][n])
            printif("Dipole: theta=%.1f, phi=%.1f" % (t*180/np.pi, p*180/np.pi), show_prints)

            random_dipole = dipole.Dipole(p, t, lda_exc=wavelength)
            self.dipole_ensemble.append(random_dipole)

    def generate_photoselection(self, dipole_count, wavelength=500e-9, show_prints=False):
        """ currently only acceping one excitation polarisation """
        # probability of excitaiton proportional to intensity 
        raise NotImplementedError("Photoselection of dipole distribution not implemented")

    def calculate_pupil_radiation(self, NA, r=1):
        """ incoherent addition (intensity) across curved pupil """
        total_intensity_x = []
        total_intensity_y = []
        angles = tuple()  # same for all dipoles
        for n in range(len(self.dipole_ensemble)):
            current_dipole = self.dipole_ensemble[n]
            angles, vals_efield_x, vals_efield_y = \
                current_dipole.generate_pupil_field(NA, r)
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
            plt.plot(pdf_x_points, pdf(plot_x)/norm)
        return accepted_points
