import numpy as np
from matplotlib import pyplot as plt
import scipy

## functions for matrices that need to me multiplied together to give tracing for
# E-field and k-vector

def meridional_transform(phi, inverse=False):
    merid_transform = np.array([
        [np.cos(phi), np.sin(phi), 0],
        [-np.sin(phi), np.cos(phi), 0],
        [0, 0, 1]
    ])
    if inverse:
        try:
            merid_transform = np.linalg.inv(merid_transform)
        except:
            print(merid_transform)
            raise Exception("COULD NOT INVERT?")
            
    return merid_transform

def arbitrary_rotation(theta, ux,uy,uz):
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    rotate = np.array([
        [cos_t+ux**2*(1-cos_t), ux*uy*(1-cos_t)-uz*sin_t, ux*uz*(1-cos_t)+uy*sin_t],
        [uy*ux*(1-cos_t)+uz*sin_t, cos_t+uy**2*(1-cos_t), uy*uz*(1-cos_t)-ux*sin_t],
        [uz*ux*(1-cos_t)-uy*sin_t, uz*uy*(1-cos_t)+ux*sin_t, cos_t + uz**2*(1-cos_t)]
    ])
    if hasattr(ux, "__len__") and len(ux) > 1:
        rotate = np.swapaxes(rotate,0,2)
        # rotate = rotate.reshape((rotate.shape[2], 3, 3))
    return rotate

def refraction_meridional(theta):
    refract = np.array([
        [np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)]
    ])
    return refract

def rotate_meridional(theta):
    return refraction_meridional(theta)

def rotate_basis(phi, theta, inverse=False):
    """ 
    Called by _rotate_field_curved_surface
    changes coordinate system according to k vector so that Iz = 0
    both rotates into meridional and then does theta rotation
    """
    rotate = np.array([
        [np.cos(phi)*np.cos(theta), np.sin(phi)*np.cos(theta), -np.sin(theta)],
        [-np.sin(phi), np.cos(phi), 0],
        [np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)]
    ])
    if inverse:
        try:
            rotate = np.linalg.inv(rotate)
        except:
            print(rotate)
            raise Exception("COULD NOT INVERT ROTATE BASIS VECTOR")
    return rotate

def polariser(psi):
    polariser = np.array([
        [np.cos(psi)**2, np.sin(psi)*np.cos(psi), 0],
        [np.sin(psi)*np.cos(psi), np.sin(psi)*np.sin(psi), 0],
        [0, 0, 1]
    ])
    return polariser

## add frensel coord rotation for p and s
def ps_wave_transform(theta):
    pass

def mirror_rotation_transform(y_rot):
    pass
    # need to work this out, incorporate with the p/s wave transform already in literature?

def fresnel_mirror_flat(rp, rs):
    mirror = np.array([
        [rp, 0, 0],
        [0, rs, 0],
        [0, 0, 1]
    ])
    return mirror

def fresnel_matrix(theta_i, n1, n2, reflection=True):
    sin_theta_t = (n1/n2)*np.sin(theta_i)
    print("sin_theta_t", sin_theta_t)
    print("n2/n1", n2/n1)
    if n1>n2 and sin_theta_t >= 1:
        print("Total internal reflection!")
        theta_t = np.pi/2
    else:
        theta_t = np.arcsin(sin_theta_t)

    print("theta_i", theta_i, theta_i*180/np.pi, "theta_t", theta_t, theta_t*180/np.pi)
    r_s = (n1*np.cos(theta_i) - n2*np.cos(theta_t))/\
        (n1*np.cos(theta_i) + n2*np.cos(theta_t))
    r_p = (n2*np.cos(theta_i) - n1*np.cos(theta_t))/\
        (n2*np.cos(theta_i) + n1*np.cos(theta_t))
    
    t_s = 2*n1*np.cos(theta_i)/(n1*np.cos(theta_i)+n2*np.cos(theta_t))
    t_p = 2*n1*np.cos(theta_i)/(n2*np.cos(theta_i)+n1*np.cos(theta_t))

    mat_t = np.array([
        [t_p, 0, 0],
        [0, t_s, 0],
        [0, 0, 1]
    ])
    mat_r = np.array([
        [r_p, 0, 0],
        [0, r_s, 0],
        [0, 0, 1]
    ])
    if reflection:
        return mat_r
    else:
        return mat_t, theta_t

def wave_plate(psi, delta):
    waveplate = np.array([
        [np.cos(delta/2) + 1j*np.cos(2*psi)*np.sin(delta/2), 1j*np.sin(2*psi)*np.sin(delta/2), 0],
        [1j*np.sin(2*psi)*np.sin(delta/2), np.cos(delta/2) - 1j*np.cos(2*psi)*np.sin(delta/2), 0],
        [0, 0, 1]
    ])
    return waveplate

def diagonal(value=1):
    return value*np.identity(3)

def ps_projection_matrix(p, s, k_vec, inverse=False):
    """project E onto p and s component in reflection, p = k1 x N, s = k1 x p"""
    # for loop instead of vectorise just for safety.. reshape likes to muddle up elements
    ps_projection = np.zeros((p.shape[0], 3, 3))

    if k_vec.ndim < 2:
        k_vec = np.expand_dims(k_vec, 0)
    if p.ndim < 2:
        p = np.expand_dims(p, 1)

    print(p.shape)
    print(p[0,0].shape)
    print(k_vec[0,0].shape)
    print(k_vec.shape)

    for n in range((p.shape[0])):

        if(np.all(np.equal(p[n,:], 0))):
            ps_projection[n, :, :] = np.identity(3)
        else:
            ps_projection[n, :, :] = np.array([
                [p[n,0], p[n,1], p[n,2]],
                [s[n,0], s[n,1], s[n,2]],
                [k_vec[n,0], k_vec[n,1], k_vec[n,2]]
            ])
        if inverse:
            ps_projection[n, :, :] = np.linalg.inv(ps_projection[n, :, :])
    return ps_projection

# def ps_projection_matrix(p, s, k_vec):
#     ps_projection_mat = np.array([
#         [p[0], p[1], p[2]],
#         [s[0], s[1], s[2]],
#         [k_vec[0], k_vec[1], k_vec[2]]
#     ])
#     return ps_projection_mat

def flip_axis(axis=2):
    flip = np.identity(3)
    flip[axis,axis] = -1
    return flip

def rotate_rays_x(angle):
    rot_mat_y = np.array([
        [np.cos(angle), 0, np.sin(angle)],
        [0, 1, 0],
        [-np.sin(angle), 0, np.cos(angle)]
    ])
    rot_mat = np.array([
        [1, 0, 0],
        [0, np.cos(angle), -np.sin(angle)],
        [0, np.sin(angle), np.cos(angle)]
    ])
    return rot_mat

def rotate_rays_y(angle):
    rot_mat_y = np.array([
        [np.cos(angle), 0, np.sin(angle)],
        [0, 1, 0],
        [-np.sin(angle), 0, np.cos(angle)]
    ])
    return rot_mat_y

def reflection_cartesian_matrix(N):
    # householder transformation, may need to replace - with +
    reflect_matrix = [
        [1-2*N[0]**2, -2*N[0]*N[1], -2*N[0]*N[2]],
        [-2*N[0]*N[1], 1-2*N[1]**2, -2*N[1]*N[2]],
        [-2*N[0]*N[2], -2*N[1]*N[2], 1-2*N[2]**2]
    ]
    return reflect_matrix

def single_suface_fresnel_matrix(theta_i, n_substrate_data, wavelength=500e-9,
                                    plot_debug=False, reflection=True):
    n_complex, _, _ = interpolate_n_from_data(theta_i, n_substrate_data, wavelength)

    r_p, r_s, transmission_angle = \
        compute_fresnel_single_surface(theta_i, n_complex, reflection)

    fresnel_matrix = np.zeros([np.size(r_p), 3, 3], dtype=np.complex64)

    for n in range(np.size(r_p)):
        fresnel_matrix[n, :, :] = np.array([
        [r_p[n], 0, 0],
        [0, r_s[n], 0],
        [0, 0, 1]
    ])
    if not reflection:
        return fresnel_matrix, transmission_angle
    else:
        return fresnel_matrix

def interpolate_n_from_data(theta_i, n_data, wavelength):
    if hasattr(n_data, "__len__"):
        wl_mul=1e9
        n_wl = n_data[:,0]/wl_mul
        n = n_data[:,1]
        k = n_data[:,2]

        n_interp = scipy.interpolate.interp1d(n_wl, n)
        k_interp = scipy.interpolate.interp1d(n_wl, k)

        if hasattr(wavelength, "__len__"):
            if len(wavelength == 1):
                wavelength = np.ones(len(theta_i))*wavelength

        wl_test = np.arange(300,800)/wl_mul
        # if just one wavelength, to visualise for debugging
        n_complex_test = n_interp(wl_test) + 1j*k_interp(wl_test)

        n_complex = n_interp(wavelength) + 1j*k_interp(wavelength)
        print("n_complex", n_complex)
    return n_complex, wl_test, n_complex_test

def compute_fresnel_single_surface(theta_1, n_surface, reflection=True):
    n1 = 1
    n2 = n_surface

    sin_theta1 = np.sin(theta_1)
    cos_theta1 = np.cos(theta_1)
    sin_theta2 = np.sin(theta_1)*n1/n2
    cos_theta2 = (1 - sin_theta2**2)**0.5

    # reflection from first surface
    r_12p = (n1/cos_theta1 - n2/cos_theta2)/(n1/cos_theta1 + n2/cos_theta2)
    r_12s = (n1*cos_theta1 - n2*cos_theta2)/(n1*cos_theta1 + n2*cos_theta2)

    # transmission through first surface
    t_12p = (2*n1/cos_theta1)/(n1/cos_theta1 + n2/cos_theta2)
    t_12s = (2*n1*cos_theta1)/(n1*cos_theta1 + n2*cos_theta2)

    transmission_angle = np.arccos(cos_theta2)

    if reflection:
        return np.squeeze(r_12p), np.squeeze(r_12s), transmission_angle
    else:
        return np.squeeze(t_12p), np.squeeze(t_12s), transmission_angle
    

def thin_film_fresnel_matrix(theta_i, n_film_data, film_thickness,
                                    n_substrate_data, wavelength=500e-9,
                                    plot_debug=False, reflection=True):
    """get fresnel matrix for thin-film protected mirror"""
    
    # TODO replace with interpolate_n_from_data
    if hasattr(n_film_data, "__len__") and hasattr(n_substrate_data, "__len__"):
        wl_mul=1e9
        n_film_wl = n_film_data[:,0]/wl_mul
        n_film = n_film_data[:,1]
        k_film = n_film_data[:,2]

        n_substrate_wl = n_substrate_data[:,0]/wl_mul
        n_substrate = n_substrate_data[:,1]
        k_substrate = n_substrate_data[:,2]

        n_film_interp = scipy.interpolate.interp1d(n_film_wl, n_film)
        k_film_interp = scipy.interpolate.interp1d(n_film_wl, k_film)

        n_substrate_interp = scipy.interpolate.interp1d(n_substrate_wl, n_substrate)
        k_substrate_interp = scipy.interpolate.interp1d(n_substrate_wl, k_substrate)

        # print("theta_i")

        if hasattr(wavelength, "__len__"):
            if len(wavelength == 1):
                wavelength = np.ones(len(theta_i))*wavelength

        wl_test = np.arange(300,800)/wl_mul
        n_film_test = n_film_interp(wl_test) + 1j*k_film_interp(wl_test)
        n_substrate_test =  n_substrate_interp(wl_test) + 1j*k_substrate_interp(wl_test)

        n_film_complex = n_film_interp(wavelength) + 1j*k_film_interp(wavelength)
        n_substrate_complex = n_substrate_interp(wavelength) + 1j*k_substrate_interp(wavelength)

        print("n_film_complex", n_film_complex)
        print("n_substrate_complex",n_substrate_complex)

        if plot_debug:
            plt.figure()
            plt.plot(wl_test, n_film_test.real, label=r'$\mathcal{R}(n_{film})$')
            plt.plot(wl_test, n_film_test.imag, label=r'$\mathcal{I}(n_{film})$')
            plt.plot(wl_test, np.abs(n_film_test), label=r'$|n_{film}|$')
            plt.xlabel("Wavelength")
            plt.legend()
            plt.show()
            plt.figure()
            plt.plot(wl_test, n_substrate_test.real, label=r'$\mathcal{R}(n_{metal})$')
            plt.plot(wl_test, n_substrate_test.imag, label=r'$\mathcal{I}(n_{metal})$')
            plt.plot(wl_test, np.abs(n_substrate_test), label=r'$|n_{metal}|$')
            plt.xlabel("Wavelength")
            plt.legend()
            plt.show()

    else:
        n_film_complex=n_film_data
        n_substrate_complex=n_substrate_data
    
    r_p, r_s, transmission_angle = compute_fresnel_thin_film(theta_i, n_film_complex, film_thickness,
                                        n_substrate_complex, wavelength, reflection)
    # print("r_p", r_p)
    # print("r_s", r_s)
    # print(r_p.shape)
    # print(r_s.shape)
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


def compute_fresnel_thin_film(theta_1, n_film, d, n_metal,
                                      wavelength, reflection=True):
    n1 = 1
    n2 = n_film
    n3 = n_metal

    sin_theta1 = np.sin(theta_1)
    cos_theta1 = np.cos(theta_1)
    sin_theta2 = np.sin(theta_1)*n1/n2
    cos_theta2 = (1 - sin_theta2**2)**0.5

    # reflection from first surface
    r_12p = (n1/cos_theta1 - n2/cos_theta2)/(n1/cos_theta1 + n2/cos_theta2)
    r_12s = (n1*cos_theta1 - n2*cos_theta2)/(n1*cos_theta1 + n2*cos_theta2)

    # transmission through first surface
    t_12p = (2*n1/cos_theta1)/(n1/cos_theta1 + n2/cos_theta2);
    t_12s = (2*n1*cos_theta1)/(n1*cos_theta1 + n2*cos_theta2);

    sin_theta3 = sin_theta2*n2/n3
    cos_theta3 = (1 - sin_theta3**2)**0.5

    # reflection from second surface (e.g. film-silver interface)
    r_2p = (n2/cos_theta2 - n3/cos_theta3)/(n2/cos_theta2 + n3/cos_theta3);
    r_2s = (n2*cos_theta2 - n3*cos_theta3)/(n2*cos_theta2 + n3*cos_theta3);

    # transmission through second surface 
    t_2p = (2*n1/cos_theta2)/(n1/cos_theta2 + n2/cos_theta3);
    t_2s = (2*n1*cos_theta2)/(n1*cos_theta2 + n2*cos_theta3);

    beta_complex = (2*np.pi/wavelength)*d*n2*cos_theta2

    # from Tompkins Irene
    rp_total = (r_12p + r_2p*np.exp(1j*2.*beta_complex))/(1+r_12p*r_2p*np.exp(1j*2.*beta_complex))
    rs_total = (r_12s + r_2s*np.exp(1j*2.*beta_complex))/(1+r_12s*r_2s*np.exp(1j*2.*beta_complex))

    tp_total = (t_12p + t_2p*np.exp(1j*2.*beta_complex))/(1+t_12p*t_2p*np.exp(1j*2.*beta_complex))
    ts_total = (t_12s + t_2s*np.exp(1j*2.*beta_complex))/(1+t_12s*r_2s*np.exp(1j*2.*beta_complex))

    transmission_angle = np.arccos(cos_theta3)

    if reflection:
        return np.squeeze(rp_total), np.squeeze(rs_total), transmission_angle
    else:
        return np.squeeze(tp_total), np.squeeze(ts_total), transmission_angle

def compute_fresnel_protected_mirror_old(theta_1, n_film, d, n_metal, wavelength):
    if hasattr(n_film, "__len__"):
        n_points = len(n_film)
    else:
        n_points = 1
    n1 = np.ones(n_points)
    n2 = n_film
    n3 = n_metal

    print("n1", n1)
    print("n2", n2)
    print("n3", n3)

    theta_t2 = np.arcsin((n1/n2)*np.sin(theta_1))

    r_1s = (n1*np.cos(theta_1) - n2*np.cos(theta_t2))/\
        (n1*np.cos(theta_1) + n2*np.cos(theta_t2))

    r_1p = (n2*np.cos(theta_1) - n1*np.cos(theta_t2))/\
        (n2*np.cos(theta_1) + n1*np.cos(theta_t2))

    t_1s = 2*n1*np.cos(theta_1)/(n1*np.cos(theta_1) + n2*np.cos(theta_t2));
    t_1p = 2*n1*np.cos(theta_1)/(n2*np.cos(theta_1) + n1*np.cos(theta_t2));

    # n3 is complex
    theta_t3 = np.arcsin((n2/n3)*np.sin(theta_t2));
    theta_r3 = theta_t2; 

    r_2s = (n2*np.cos(theta_t2) - n3*np.cos(theta_t3))/\
        (n2*np.cos(theta_t2) + n3*np.cos(theta_t3));
    r_2p = (n3*np.cos(theta_t2) - n2*np.cos(theta_t3))/\
        (n3*np.cos(theta_t2) + n2*np.cos(theta_t3));

    r_12s = (n2*np.cos(theta_t2) - n1*np.cos(theta_1))/\
        (n2*np.cos(theta_t2) + n1*np.cos(theta_1))
    r_12p = (n1*np.cos(theta_t2) - n2*np.cos(theta_1))/\
        (n1*np.cos(theta_t2) + n2*np.cos(theta_1))

    # now going from sio2 to air

    t_21s = 2*n2*np.cos(theta_t2)/(n2*np.cos(theta_t2) + n1*np.cos(theta_1));
    t_21p = 2*n2*np.cos(theta_t2)/(n1*np.cos(theta_t2) + n2*np.cos(theta_1));

    beta = 2*np.pi*d/wavelength*np.real(n2)*np.cos(theta_t2);

    rp_total = r_1p + t_1p*t_21p*r_2p*np.exp(-1j*2*beta)/(1-r_2p*r_12p*np.exp(-1j*2*beta));
    rs_total = r_1s + t_1s*t_21s*r_2s*np.exp(-1j*2*beta)/(1-r_2s*r_12s*np.exp(-1j*2*beta));

    return np.squeeze(rp_total), np.squeeze(rs_total)

## since each ray has different phi and theta, need a tensor version of these 
## matrices to correctly transform all rays at once

def rotate_basis_tensor(phi_array, theta_array, inverse=False):
    rotate_tensor = np.zeros((len(phi_array), 3, 3))
    for n in range(len(phi_array)):
        rotate_tensor[n, :, :] = rotate_basis(
            phi_array[n].squeeze(), theta_array[n].squeeze(),
            inverse)
    return rotate_tensor

def meridional_transform_tensor(phi_array, inverse=False):
    meridional_tensor = np.zeros((len(phi_array), 3, 3))
    for n in range(len(phi_array)):
        meridional_tensor[n, :, :] = meridional_transform(
            phi_array[n].squeeze(), inverse)
    return meridional_tensor

def refraction_meridional_tensor(theta_array):
    # after putting rays into meridional basis (phi_m=0) do refraction
    refract_tensor = np.zeros((len(theta_array), 3, 3))
    for n in range(len(theta_array)):
        refract_tensor[n, :, :] = refraction_meridional(
            theta_array[n].squeeze())
    return refract_tensor

