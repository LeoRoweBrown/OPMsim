import numpy as np
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

def fresnel_matrix(theta_i, n1, n2):
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
    mat = np.array([
        [r_p, 0, 0],
        [0, r_s, 0],
        [0, 0, 1]
    ])
    return mat

def wave_plate(psi, delta):
    waveplate = np.array([
        [np.cos(delta/2) + 1j*np.cos(2*psi)*np.sin(delta/2), 1j*np.sin(2*psi)*np.sin(delta/2), 0],
        [1j*np.sin(2*psi)*np.sin(delta/2), np.cos(delta/2) - 1j*np.cos(2*psi)*np.sin(delta/2), 0],
        [0, 0, 1]
    ])
    return waveplate

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

def protected_mirror_fresnel_matrix(theta_i, n_film_data, film_thickness, n_metal_data, wavelength=500e-9):
    wl_mul=1e9
    n_film_wl = n_film_data[:,0]/wl_mul
    n_film = n_film_data[:,1]
    k_film = n_film_data[:,2]

    n_metal_wl = n_metal_data[:,0]/wl_mul
    n_metal = n_metal_data[:,1]
    k_metal = n_metal_data[:,2]

    n_film_interp = scipy.interpolate.interp1d(n_film_wl, n_film)
    k_film_interp = scipy.interpolate.interp1d(n_film_wl, k_film)

    n_metal_interp = scipy.interpolate.interp1d(n_metal_wl, n_metal)
    k_metal_interp = scipy.interpolate.interp1d(n_metal_wl, k_metal)

    # print("theta_i")

    if hasattr(wavelength, "__len__"):
        if len(wavelength == 1):
            wavelength = np.ones(len(theta_i))*wavelength

    n_film_complex = n_film_interp(wavelength) - 1j*k_film_interp(wavelength)
    n_metal_complex = n_metal_interp(wavelength) - 1j*k_metal_interp(wavelength)
    
    r_p, r_s = compute_fresnel_protected_mirror(theta_i, n_film_complex, film_thickness, n_metal_complex, wavelength)
    # print("r_p", r_p)
    # print("r_s", r_s)
    # print(r_p.shape)
    # print(r_s.shape)
    
    fresnel_matrix = np.zeros([np.size(r_p), 3, 3], dtype=np.complex64)

    for n in range(np.size(r_p)):
        # print(r_p[n])
        fresnel_matrix[n, :, :] = np.array([
        [r_p[n], 0, 0],
        [0, r_s[n], 0],
        [0, 0, 1]
    ])
    print("fresnel_matrix.shape", fresnel_matrix.shape)

    return fresnel_matrix

def compute_fresnel_protected_mirror(theta_1, n_film, d, n_metal, wavelength):
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

