from matplotlib import pyplot as plt
import numpy as np
import warnings
from . import distribution_functions
from .rays import PolarRays
from .tools.misc import printif
from . import matrices
from opmsim.visualization.dipole_source_plots import plot_ray_sphere
from opmsim.visualization.dipole_source_plots import plot_points_on_sphere, plot_dipole_source_3d

class DipoleSource:
    """
    Source made of multiple dipoles  # TODO redo docstring

    """

    def __init__(self, dipole_orientations=(()), name="default source", lda_exc=500.e-9, lda_em=500.e-9):
        """Constructor populates dipole source with dipoles passed by Nx2 array of dipole orientations

        Args:
            dipole_orientations (tuple, optional): Nx2 array of dipole orientations (phi_d, alpha_d). Defaults to (()).
            name (str, optional): source label/name. Defaults to "default source".
            lda_exc (float, optional): Excitation wavelength. Unused. Defaults to 500.e-9.
            lda_em (float, optional): Emission wavelength. Defaults to 500.e-9.
        """

        phi_alpha_pairs = np.array(dipole_orientations).reshape((-1, 2))  # -1 to infer length, 2D for phi, alpha
        phi_d_array = np.array(phi_alpha_pairs[:, 0])
        alpha_d_array = np.array(phi_alpha_pairs[:, 1])

        self.alpha_d = alpha_d_array
        self.phi_d = phi_d_array
        self.lda_em = lda_em
        self.lda_exc = lda_exc
        self.emission_scaling = np.ones(
            (len(alpha_d_array), 1, 1, 1)
        )  # scales quantity/emission_scaling/fractional quantity of dipole

        self.n_dipoles = len(phi_d_array)

        self.excitation_polarisation = None
        self.rays = PolarRays([], [])

        self._get_p_vec()

    def _get_p_vec(self):
        self.p_vec = np.column_stack(
            (np.cos(self.alpha_d) * np.cos(self.phi_d), np.cos(self.alpha_d) * np.sin(self.phi_d), np.sin(self.alpha_d))
        )
        self.p_vec = self.p_vec.reshape((len(self.phi_d), 3))

    def get_initial_e_fields(self, rays: PolarRays):
        # use _get_e_field to calculate the E field based on a propagation vector
        # and the dipole distribution
        n_vec = rays.k_vec
        p_vec = self.p_vec

        # r = rays.radius DEPRECATED the radius attrib TODO: tidy up
        r = 1

        n_dipoles = p_vec.shape[0]
        n_rays = n_vec.shape[0]

        # prepare to broadcast, index=0: dipoles index=1:n_rays
        p_vec = np.reshape(p_vec, (n_dipoles, 1, 3))
        n_vec = np.reshape(n_vec, (1, n_rays, 3))

        n_x_p = np.cross(n_vec, p_vec)
        k = 2 * np.pi / self.lda_exc

        rays.e_field = np.cross(n_x_p, n_vec)  # (N_rays, N_dipole, 3)
        rays.e_field = rays.e_field.reshape((n_dipoles, n_rays, 3, 1))  # e_fields are (3x1)
        rays.e_field_pre = (np.e ** (1j * k * r) / r) * k**2  # replace with distribution of k (lambda_exc)
        self.emission_scaling = self.emission_scaling.reshape(n_dipoles, 1, 1, 1)
        rays.emission_scaling = self.emission_scaling  # apply to rays object so calculate_intensity works
        # TODO totally move this to rays, do not store in DipoleSource object?

        # Get initial energy calculations
        rays.calculate_intensity()

    def add_dipoles(self, dipole_angles):
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

    def generate_dipole_ensemble(
            self, dipole_count, lda_ex=500e-9, lda_em=500e-9, show_prints=False, plot=False):
        """
        Generate uniformly distriubted dipoles with same wavelength
        doesn't support beta, slow tumbling etc.
        """

        phi_d, theta_d, areas = distribution_functions.fibonacci_dipole_generation(
            point_count=dipole_count)
        self.phi_d = np.append(self.phi_d, phi_d)
        self.alpha_d = np.append(self.alpha_d, np.pi / 2 - theta_d)

        self.n_dipoles = len(self.phi_d)
        self.emission_scaling = np.ones(len(self.phi_d))

        # plot to verify distribution
        if plot:
            self.plot_distribution()

        printif("Generating %d dipoles" % self.n_dipoles, show_prints)
        self._get_p_vec()

    def depolarise(self, correlation_time, fluorescence_lifetime, timepoints=100):
        raise NotImplementedError()

    def plot_distribution(self, alphas=[], show_plot=True, plot_3d=False):
        """Plot dipole distribution on sphere and return mpl figure"""
        if plot_3d:
            return plot_dipole_source_3d(
                self.alpha_d, self.phi_d, alphas=self.emission_scaling,
                directional_arrow=self.excitation_polarisation,
                show_plot=True, dipole_style='arrow')
        else:
            return plot_points_on_sphere(
                self.alpha_d, self.phi_d, self.emission_scaling, style='arrow', show_plot=show_plot)
        # raise NotImplementedError("Moved to visualization package!")

    def classical_photoselection(self, excitation_polarisation, plot=False):
        """Scale the intensity from dipoles based on their orientation and the
        excitation polarisation"""

        if excitation_polarisation is None:
            if plot:
                alphas = np.ones_like(self.phi_d) * 1 / 3  # a little opacity (33%)
                self.plot_distribution(alphas)
            return
        dipole_count = len(self.phi_d)
        phi_exc, alpha_exc = excitation_polarisation
        self.emission_scaling = np.ones(len(self.phi_d))
        self.excitation_polarisation = excitation_polarisation

        phi_d = self.phi_d
        alpha_d = self.alpha_d
        cos_d_exc = (
            np.cos(alpha_exc) * np.cos(phi_exc) * np.cos(alpha_d) * np.cos(phi_d)
            + np.cos(alpha_exc) * np.sin(phi_exc) * np.cos(alpha_d) * np.sin(phi_d)
            + np.sin(alpha_exc) * np.sin(alpha_d)
        )
        self.emission_scaling = cos_d_exc**2  # normalise to 1

        if plot:
            alphas = self.emission_scaling
            self.plot_distribution(alphas)

    def display_pupil_rays(self):
        warnings.warn("Moved display_pupil_rays to visualization.dipole_source_plots", DeprecationWarning)

    def get_rays_uniform(
            self, max_half_angle, f,
            ray_count=5000, plot_sphere=False, generation_method="fibonacci",
            ring_method="uniform_phi_inbetween"):
        """Get equal area elements in rings for uniform rays, also compute their area"""
        print("Generating rays")
        if generation_method == "rings":
            phi_k, theta_k, areas = distribution_functions.uniform_points_on_sphere(
                max_half_angle, ray_count, ring_method)
        elif generation_method == "fibonacci":
            phi_k, theta_k, areas = distribution_functions.fibonacci_ray_generation(
                max_half_angle, ray_count)
        else:
            raise Exception(f"Invalid ray distribution method '{generation_method}'")
        if plot_sphere:
            plot_ray_sphere(phi_k, theta_k)

        self.max_half_angle = max_half_angle
        self.ray_area = areas  # do we need dipole_source to have this attribute?

        self.rays = PolarRays(phi_k, theta_k, f, areas, lda=self.lda_em)
        self.get_initial_e_fields(self.rays)

    def define_custom_rays(self, phi_k, theta_k, Ex, Ey, Ez):
        """
        Define custom rays for testing (comparison to CWD calculations)
        """
        print("Defining custom rays, remember to add custom_rays=True to options when running OpticalSystem!")
        self.add_dipoles((0, 0))  # dummy dipole
        n_rays = len(phi_k)
        areas = np.array([1] * n_rays)

        rays = PolarRays(phi_k, theta_k, None, areas, lda=self.lda_em)

        n_dipoles = 1
        print("Ex", Ex)
        print("Ey", Ey)
        print("Ez", Ez)

        rays.e_field = np.zeros((1, n_rays, 3, 1))
        rays.e_field[0, :, :, 0] = np.vstack([Ex, Ey, Ez]).T

        max_abs_e = np.max(np.abs(rays.e_field))
        dot_product = abs(np.sum(rays.e_field * rays.k_vec, axis=2))
        if dot_product > 1e-9 * max_abs_e:
            print("dot product", np.sum(rays.e_field * rays.k_vec, axis=2))
            raise Exception("Supplied electric field and wavevector invalid: dot product must be zero")

        self.emission_scaling = self.emission_scaling.reshape(n_dipoles, 1, 1, 1)
        print(self.emission_scaling.shape)
        rays.total_intensity_initial = np.sum(rays.e_field * rays.e_field * self.emission_scaling, axis=0)
        # energy per dipole
        rays.calculate_intensity()

        self.rays = rays

    def simulate_rotational_diffusion(self, p_vec, D, tau, timepoints=None):
        """
        Return new dipole orientations after depolarisation from rotational diffusion
        """
        raise NotImplementedError
