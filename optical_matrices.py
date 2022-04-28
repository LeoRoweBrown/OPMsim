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