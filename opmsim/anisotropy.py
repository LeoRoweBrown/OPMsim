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

def calculate_anisotropy_detector(detector):
    """a bit pointless and over-engineered"""
    I_p = detector.Iy_integral
    I_s = detector.Ix_integral

    r = (I_p - I_s)/(I_p + 2*I_s)
    return r

def calculate_anisotropy(I_p, I_s):

    r = (I_p - I_s)/(I_p + 2*I_s)
    return r

def theoretical_anisotropy(NA, dipole_orientation, return_intensities=False):
    rho_0 = np.arcsin(NA)  # assume n=1
    cos_rho = np.cos(rho_0)
    K_a = (2 - 3*cos_rho + cos_rho*cos_rho*cos_rho)/3
    K_b = (1 - 3*cos_rho + 3*cos_rho*cos_rho - cos_rho*cos_rho*cos_rho)/12
    K_c = (5 - 3*cos_rho - cos_rho*cos_rho - cos_rho*cos_rho*cos_rho)/4

    # dipole angles into cartesian norm vector
    phi, alpha = dipole_orientation
    print("alpha:", alpha)
    print("phi:", phi)
    x = [ np.sin(alpha), np.cos(alpha)*np.sin(phi), \
        np.cos(alpha)*np.cos(phi)]

    I_s = K_a*x[0]*x[0] + K_c*x[1]*x[1] + K_b*x[2]*x[2]
    I_p = K_a*x[0]*x[0] + K_b*x[1]*x[1] + K_c*x[2]*x[2]

    r = (I_p - I_s)/(I_p + 2*I_s)

    if return_intensities:
        return r, I_p, I_s
    else:
        return r

def theoretical_anisotropy_population(NA, return_intensities=False):
    """only correct for dipole in x-orientation, TODO: generalise this at some point
    from https://doi.org/10.1016/S0006-3495(79)85271-6"""
    # assume n=1
    cos_rho = np.sqrt(1-NA**2)
    # print(cos_rho)
    K_a = (2 - 3*cos_rho + cos_rho*cos_rho*cos_rho)/3
    K_b = (1 - 3*cos_rho + 3*cos_rho*cos_rho - cos_rho*cos_rho*cos_rho)/12
    K_c = (5 - 3*cos_rho - cos_rho*cos_rho - cos_rho*cos_rho*cos_rho)/4

    # result from integration
    I_p = (2/15)*(K_b + 3*K_c + K_a)
    I_s = (2/15)*(3*K_b + K_c + K_a)

    r = (I_p - I_s)/(I_p + 2*I_s)

    if return_intensities:
        return r, I_p, I_s
    else:
        return r