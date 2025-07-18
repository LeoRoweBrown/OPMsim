import numpy as np

def calculate_anisotropy_rawdata(detector_p, detector_s, simulate_polariser=False):

    if simulate_polariser:
        I_p = detector_p.Iy_integral
        I_s = detector_s.Ix_integral
    else:
        I_p = detector_p.I_total_integral
        I_s = detector_s.I_total_integral

    r = (I_p - I_s)/(I_p + 2*I_s)
    return r

def calculate_anisotropy(detector, polariser_phi=None):
    """a bit pointless and over-engineered"""
    if polariser_phi is None:
        I_p = detector.Iy_integral
        I_s = detector.Ix_integral
    else:
        I_p = detector.Ix_integral*np.cos(polariser_phi) +\
            detector.Iy_integral*np.sin(polariser_phi)
        I_s = detector.Ix_integral*np.sin(polariser_phi) +\
            detector.Iy_integral*np.cos(polariser_phi)

    r = (I_p - I_s)/(I_p + 2*I_s)
    return r

def calculate_anisotropy_IpIs(I_p, I_s):
    r = (I_p - I_s)/(I_p + 2*I_s)
    return r

def theoretical_anisotropy(NA, dipole_orientation, I_p_yaxis=True, return_intensities=False):
    # calculate theoretical anisotropy for single dipole as a function of NA,
    # parallel axis is y
    rho_0 = np.arcsin(NA)  # assume n=1
    cos_rho = np.cos(rho_0)

    K_a = (2 - 3*cos_rho + cos_rho*cos_rho*cos_rho)/3
    K_b = (1 - 3*cos_rho + 3*cos_rho*cos_rho - cos_rho*cos_rho*cos_rho)/12
    K_c = (5 - 3*cos_rho - cos_rho*cos_rho - cos_rho*cos_rho*cos_rho)/4


    # dipole angles into cartesian norm vector
    phi, alpha = dipole_orientation
    print("alpha:", alpha)
    print("phi:", phi)
    # in Axelrod paper, x1 is optic axis and x3 is parallel to excitation 
    # x = [ np.sin(alpha), np.cos(alpha)*np.sin(phi), \
    #     np.cos(alpha)*np.cos(phi)]
    x = [ np.sin(alpha), np.cos(alpha)*np.cos(phi), \
        np.cos(alpha)*np.sin(phi)]
    print("x",x)
    print("K_a", K_a, "K_b", K_b, "K_c", K_c)


    I_s = K_a*x[0]*x[0] + K_c*x[1]*x[1] + K_b*x[2]*x[2]
    I_p = K_a*x[0]*x[0] + K_b*x[1]*x[1] + K_c*x[2]*x[2]

    r = (I_p - I_s)/(I_p + 2*I_s)

    if return_intensities:
        return r, I_p, I_s
    else:
        return r

def theoretical_anisotropy_population(NA, return_intensities=False):
    """only correct for photoselection in x-orientation, TODO: generalise this at some point
    from https://doi.org/10.1016/S0006-3495(79)85271-6"""
    # assume n=1
    cos_rho = np.sqrt(1-NA**2)
    # print(cos_rho)
    K_a = (2 - 3*cos_rho + cos_rho*cos_rho*cos_rho)/3
    K_b = (1 - 3*cos_rho + 3*cos_rho*cos_rho - cos_rho*cos_rho*cos_rho)/12
    K_c = (5 - 3*cos_rho - cos_rho*cos_rho - cos_rho*cos_rho*cos_rho)/4

    # result from integration done in wolfram alpha
    I_p = (2/15)*(K_b + 3*K_c + K_a)
    I_s = (2/15)*(3*K_b + K_c + K_a)

    r = (I_p - I_s)/(I_p + 2*I_s)

    if return_intensities:
        return r, I_p, I_s
    else:
        return r