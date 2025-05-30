import numpy as np


def polariser(psi):
    polariser = np.array(
        [
            [np.cos(psi) ** 2, np.sin(psi) * np.cos(psi), 0],
            [np.sin(psi) * np.cos(psi), np.sin(psi) * np.sin(psi), 0],
            [0, 0, 1],
        ]
    )
    return polariser


def fresnel_mirror_flat(rp, rs):
    mirror = np.array([[rp, 0, 0], [0, rs, 0], [0, 0, 1]])
    return mirror


def wave_plate(psi, delta):
    waveplate = np.array(
        [
            [np.cos(delta / 2) + 1j * np.cos(2 * psi) * np.sin(delta / 2), 1j * np.sin(2 * psi) * np.sin(delta / 2), 0],
            [1j * np.sin(2 * psi) * np.sin(delta / 2), np.cos(delta / 2) - 1j * np.cos(2 * psi) * np.sin(delta / 2), 0],
            [0, 0, 1],
        ]
    )
    return waveplate


def lens_refraction_meridional(theta):
    refract = np.array(
        [
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)]
        ]
    )
    return refract


def diagonal(value=1):
    return value * np.identity(3)
