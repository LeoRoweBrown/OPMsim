from copy import Error
import numpy as np
import optical_matrices

def calculate_anisotropy(pupil_p, pupil_s, return_intensities=False, excitation_phi=0):
    """
    function to calculate anisotropy for given excitation direction
    requires a polariser to be used beforehand for each parallel and perpedicular
    directions - implemented in optical_systems.py
    """
    # check dims are correct
    intensity_p = pupil_p.data_total
    intensity_s = pupil_s.data_total
    r_points = pupil_p.polar_radii
    phi_points = pupil_p.polar_angles

    shape = intensity_p.shape
    if shape[0] != len(r_points):
        raise Error(
            "intensity matrix should be organised as row being "
            "different radius/sin(theta) values"
            )

    dphi = 2*np.pi/len(phi_points)  # should always be constant/linearly spaced

    I_p = 0
    I_s = 0

    for p_i in range(len(phi_points)):
        for r_i in range(len(r_points)-1):
            # trapezium rule, no need to normalise area because we take ratio
            # dA = r dr dphi
            dA = r_points[r_i+1]*(r_points[r_i+1] - r_points[r_i])*dphi

            if pupil_p.curved:
                r = (r_points[r_i+1] +  r_points[r_i])/2
                theta = np.arcsin(r)
                dA /= np.cos(theta)  # correct for the extra area due to curved surface

            dI_p = dA*(intensity_p[r_i, p_i] + intensity_p[r_i+1, p_i])/2  # trapezium rule
            dI_s = dA*(intensity_s[r_i, p_i] + intensity_s[r_i+1, p_i])/2
            I_p += dI_p
            I_s += dI_s
    
    r = (I_p - I_s)/(I_p + 2*I_s)
    if return_intensities:
        return r, I_p, I_s
    else:
        return r

def calculate_anisotropy_rawdata(detector_p, detector_s):
    I_p = detector_p.I_total_integral
    I_s = detector_s.I_total_integral
    r = (I_p - I_s)/(I_p + 2*I_s)
    return r

def theoretical_anisotropy(NA, dipole_orientation, return_intensities=False):
    rho_0 = np.arcsin(NA)  # assume n=1
    cos_rho = np.cos(rho_0)
    K_a = (2 - 3*cos_rho + cos_rho*cos_rho*cos_rho)/3
    K_b = (1 - 3*cos_rho + 3*cos_rho*cos_rho - cos_rho*cos_rho*cos_rho)/12
    K_c = (5 - 3*cos_rho - cos_rho*cos_rho - cos_rho*cos_rho*cos_rho)/4

    # dipole angles into cartesian norm vector
    alpha, phi = dipole_orientation
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
    """only correct for dipole in x-orientation, TODO: generalise this at some point"""
    # assume n=1
    cos_rho = np.sqrt(1-NA**2)
    # print(cos_rho)
    K_a = (2 - 3*cos_rho + cos_rho*cos_rho*cos_rho)/3
    K_b = (1 - 3*cos_rho + 3*cos_rho*cos_rho - cos_rho*cos_rho*cos_rho)/12
    K_c = (5 - 3*cos_rho - cos_rho*cos_rho - cos_rho*cos_rho*cos_rho)/4

    """
    from scipy.integrate import dblquad
    I_p, err = dblquad(lambda b, a: ( K_a*np.sin(b)**2*np.sin(a)**2 + K_c*np.cos(b)**2 + \
                K_b*np.sin(b)**2*np.cos(a)**2 )*np.cos(b)**2*np.sin(b),\
                0, np.pi/2, lambda b: 0, lambda b: 2*np.pi)
    print("fract err", err/I_p)

    I_s, err = dblquad(lambda b, a: ( K_a*np.sin(b)**2*np.sin(a)**2 + K_b*np.cos(b)**2 + \
                K_c*np.sin(b)**2*np.cos(a)**2 )*np.cos(b)**2*np.sin(b),\
                0, np.pi/2, lambda b: 0, lambda b: 2*np.pi)
    print("fract err", err/I_s)
    """
    I_p = (2/15)*(K_b + 3*K_c + K_a)
    I_s = (2/15)*(3*K_b + K_c + K_a)

    # print(I_p)
    # print(I_s)
    r = (I_p - I_s)/(I_p + 2*I_s)

    if return_intensities:
        return r, I_p, I_s
    else:
        return r