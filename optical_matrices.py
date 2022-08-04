import numpy as np
from numpy.lib.function_base import _meshgrid_dispatcher

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
            np.linalg.inv(merid_transform)
        except:
            print(merid_transform)
            raise Exception("COULD NOT INVERT?")
            
        merid_transform = np.linalg.inv(merid_transform)
    return merid_transform

def refraction_meridional(theta):
    refract = np.array([
        [np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)]
    ])
    return refract

def rotate_meridional(theta):
    return refraction_meridional(theta)

def rotate_basis(phi, theta):
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

def wave_plate(psi, delta):
    waveplate = np.array([
        [np.cos(delta/2) + 1j*np.cos(2*psi)*np.sin(delta/2), 1j*np.sin(2*psi)*np.sin(delta/2), 0],
        [1j*np.sin(2*psi)*np.sin(delta/2), np.cos(delta/2) - 1j*np.cos(2*psi)*np.sin(delta/2), 0],
        [0, 0, 1]
    ])
    return waveplate

def rotate_rays_x(angle):
    rot_mat = np.array([
        [np.cos(angle), 0, np.sin(angle)],
        [0, 1, 0],
        [-np.sin(angle), 0, np.cos(angle)]
    ])
    return rot_mat

## since each ray has different phi and theta, need a tensor version of these 
## matrices to correctly transform all rays at once

def rotate_basis_tensor(phi_array, theta_array):
    rotate_tensor = np.zeros((len(phi_array), 3, 3))
    for n in range(len(phi_array)):
        rotate_tensor[n, :, :] = rotate_basis(
            phi_array[n].squeeze(), theta_array[n].squeeze())
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