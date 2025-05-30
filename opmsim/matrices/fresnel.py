import numpy as np
import scipy.interpolate
from matplotlib import pyplot as plt


def single_surface_fresnel_matrix(theta_i, n_substrate_data,
                                  wavelength=500e-9, reflection=True):
    n_complex = _interpolate_n_from_data(theta_i, n_substrate_data, wavelength)

    r_p, r_s, transmission_angle = \
        _compute_fresnel_single_surface(theta_i, n_complex, reflection)

    fresnel_matrix = np.zeros([np.size(r_p), 3, 3], dtype=np.complex64)

    for n in range(np.size(r_p)):
        fresnel_matrix[n, :, :] = np.array([
            [r_p[n], 0, 0],
            [0, r_s[n], 0],
            [0, 0, 1]
        ])

    return fresnel_matrix, transmission_angle


def thin_film_fresnel_matrix(theta_i, n_film_data, film_thickness,
                             n_substrate_data, wavelength=500e-9,
                             plot_debug=False, reflection=True):
    """
    Get fresnel matrix for thin-film protected mirror, using thin film interference theory.

    :param theta_i: numpy array of incident angles
    :param n_film_data: film refractive index data against wavelength, n-by-2 np array
    :param film_thickness: thickness of film in SI units (m)
    :param n_substrate_data: substrate refractive index data against wavelength, n-by-2 np array
    :param wavelength: wavelength, currently scalar i.e., supporting a single value
    :param plot_debug: if True, plot complex RI with angle
    :param reflection: if True, Fresnel coefficients are for reflection
    :return:
    """

    n_film_complex = (theta_i, n_film_data, wavelength)
    n_substrate_complex = _interpolate_n_from_data(theta_i, n_substrate_data, wavelength)

    r_p, r_s, transmission_angle = _compute_fresnel_thin_film(theta_i, n_film_complex, film_thickness,
                                                              n_substrate_complex, wavelength, reflection)

    if plot_debug:
        plt.figure()
        plt.plot(theta_i, r_p.real, label=r'$\mathcal{R}(r_p)$')
        plt.plot(theta_i, r_p.imag, label=r'$\mathcal{I}(r_p)$')
        plt.plot(theta_i, np.abs(r_p), label=r'$|r_p|$')
        plt.xlabel("Incident angle (rad)")
        plt.legend()
        plt.show()
        plt.figure()
        plt.plot(theta_i, r_s.real, label=r'$\mathcal{R}(r_s)$')
        plt.plot(theta_i, r_s.imag, label=r'$\mathcal{I}(r_s)$')
        plt.plot(theta_i, np.abs(r_s), label=r'$|r_s|$')

        plt.xlabel("Incident angle (rad)")
        plt.legend()
        plt.show()

    fresnel_matrix = np.zeros([np.size(r_p), 3, 3], dtype=np.complex64)

    for n in range(np.size(r_p)):
        fresnel_matrix[n, :, :] = np.array([
            [r_p[n], 0, 0],
            [0, r_s[n], 0],
            [0, 0, 1]
        ])
    print("fresnel_matrix.shape", fresnel_matrix.shape)

    if not reflection:
        return fresnel_matrix, transmission_angle
    else:
        return fresnel_matrix


def _interpolate_n_from_data(theta_i, n_data, wavelength):
    if hasattr(n_data, "__len__"):
        wl_mul = 1e9
        n_wl = n_data[:, 0] / wl_mul
        n = n_data[:, 1]
        k = n_data[:, 2]

        n_interp = scipy.interpolate.interp1d(n_wl, n)
        k_interp = scipy.interpolate.interp1d(n_wl, k)

        if hasattr(wavelength, "__len__"):
            if len(wavelength) == 1:
                wavelength = np.ones(len(theta_i)) * wavelength

        n_complex = n_interp(wavelength) + 1j * k_interp(wavelength)
    else:
        Warning(
            "refractive index data is single-valued and not complex, "
            "returning input")
        return n_data
    return n_complex

def _compute_fresnel_single_surface(theta_1, n_surface, reflection=True):
    n1 = 1  # start in air
    n2 = n_surface  # first surface index

    cos_theta1 = np.cos(theta_1)
    sin_theta2 = np.sin(theta_1) * n1 / n2
    cos_theta2 = (1 - sin_theta2 ** 2) ** 0.5

    # reflection from first surface
    r_12p = (n1 / cos_theta1 - n2 / cos_theta2) / (n1 / cos_theta1 + n2 / cos_theta2)
    r_12s = (n1 * cos_theta1 - n2 * cos_theta2) / (n1 * cos_theta1 + n2 * cos_theta2)

    # transmission through first surface
    t_12p = (2 * n1 / cos_theta1) / (n1 / cos_theta1 + n2 / cos_theta2)
    t_12s = (2 * n1 * cos_theta1) / (n1 * cos_theta1 + n2 * cos_theta2)

    transmission_angle = np.arccos(cos_theta2)  # for reflection, not used

    if reflection:
        return np.squeeze(r_12p), np.squeeze(r_12s), transmission_angle
    else:
        return np.squeeze(t_12p), np.squeeze(t_12s), transmission_angle

def _compute_fresnel_thin_film(theta_1, n_film, d, n_metal,
                               wavelength, reflection=True):
    n1 = 1
    n2 = n_film
    n3 = n_metal

    sin_theta1 = np.sin(theta_1)
    cos_theta1 = np.cos(theta_1)
    sin_theta2 = np.sin(theta_1) * n1 / n2
    cos_theta2 = (1 - sin_theta2 ** 2) ** 0.5

    # reflection from first surface
    r_12p = (n1 / cos_theta1 - n2 / cos_theta2) / (n1 / cos_theta1 + n2 / cos_theta2)
    r_12s = (n1 * cos_theta1 - n2 * cos_theta2) / (n1 * cos_theta1 + n2 * cos_theta2)

    # transmission through first surface
    t_12p = (2 * n1 / cos_theta1) / (n1 / cos_theta1 + n2 / cos_theta2)
    t_12s = (2 * n1 * cos_theta1) / (n1 * cos_theta1 + n2 * cos_theta2)

    sin_theta3 = sin_theta2 * n2 / n3
    cos_theta3 = (1 - sin_theta3 ** 2) ** 0.5

    # reflection from second surface (e.g. film-silver interface)
    r_2p = (n2 / cos_theta2 - n3 / cos_theta3) / (n2 / cos_theta2 + n3 / cos_theta3)
    r_2s = (n2 * cos_theta2 - n3 * cos_theta3) / (n2 * cos_theta2 + n3 * cos_theta3)

    # transmission through second surface
    t_2p = (2 * n1 / cos_theta2) / (n1 / cos_theta2 + n2 / cos_theta3)
    t_2s = (2 * n1 * cos_theta2) / (n1 * cos_theta2 + n2 * cos_theta3)

    beta_complex = (2 * np.pi / wavelength) * d * n2 * cos_theta2

    # from Tompkins Irene
    rp_total = (r_12p + r_2p * np.exp(1j * 2. * beta_complex)) / (1 + r_12p * r_2p * np.exp(1j * 2. * beta_complex))
    rs_total = (r_12s + r_2s * np.exp(1j * 2. * beta_complex)) / (1 + r_12s * r_2s * np.exp(1j * 2. * beta_complex))

    tp_total = (t_12p + t_2p * np.exp(1j * 2. * beta_complex)) / (1 + t_12p * t_2p * np.exp(1j * 2. * beta_complex))
    ts_total = (t_12s + t_2s * np.exp(1j * 2. * beta_complex)) / (1 + t_12s * r_2s * np.exp(1j * 2. * beta_complex))

    transmission_angle = np.arccos(cos_theta3)

    if reflection:
        return np.squeeze(rp_total), np.squeeze(rs_total), transmission_angle
    else:
        return np.squeeze(tp_total), np.squeeze(ts_total), transmission_angle
