from typing import List
from numpy.typing import ArrayLike
import numpy as np


def polariser(psi: float):
    polariser = np.array(
        [
            [np.cos(psi) ** 2, np.sin(psi) * np.cos(psi), 0],
            [np.sin(psi) * np.cos(psi), np.sin(psi) * np.sin(psi), 0],
            [0, 0, 1],
        ]
    )
    return polariser


def fresnel_mirror_flat(rp: ArrayLike, rs: ArrayLike):
    rs = np.atleast_1d(rs)
    rp = np.atleast_1d(rp)
    mirror_tensor = np.zeros((len(rp), 3, 3))
    for n in range(len(rp)):
        mirror_tensor[n, :, :] = np.array([
            [rp[n], 0, 0],
            [0, rs[n], 0],
            [0, 0, 1]
        ])
    return mirror_tensor


def wave_plate(psi: float, delta: float):
    waveplate = np.array(
        [
            [np.cos(delta / 2) + 1j * np.cos(2 * psi) * np.sin(delta / 2), 1j * np.sin(2 * psi) * np.sin(delta / 2), 0],
            [1j * np.sin(2 * psi) * np.sin(delta / 2), np.cos(delta / 2) - 1j * np.cos(2 * psi) * np.sin(delta / 2), 0],
            [0, 0, 1],
        ]
    )
    return waveplate

def lens_refraction_meridional(theta: ArrayLike):
    theta = np.atleast_1d(theta)
    refract_tensor = np.zeros((len(theta), 3, 3))
    for n in range(len(theta)):
        refract_tensor[n, :, :] = np.array([
            [np.cos(theta[n]), 0, np.sin(theta[n])],
            [0, 1, 0],
            [-np.sin(theta[n]), 0, np.cos(theta[n])]
        ])
    return refract_tensor

def diagonal(value=1):
    return value * np.identity(3)
