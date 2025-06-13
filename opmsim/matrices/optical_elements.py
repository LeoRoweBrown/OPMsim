from typing import List
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


def fresnel_mirror_flat(rp: List[np.complex128], rs: List[np.complex128]):
    mirror = np.array([
        [rp, 0, 0],
        [0, rs, 0],
        [0, 0, 1]
    ]).swapaxes(-1, 0)
    return mirror


def wave_plate(psi: float, delta: float):
    waveplate = np.array(
        [
            [np.cos(delta / 2) + 1j * np.cos(2 * psi) * np.sin(delta / 2), 1j * np.sin(2 * psi) * np.sin(delta / 2), 0],
            [1j * np.sin(2 * psi) * np.sin(delta / 2), np.cos(delta / 2) - 1j * np.cos(2 * psi) * np.sin(delta / 2), 0],
            [0, 0, 1],
        ]
    )
    return waveplate


def lens_refraction_meridional(theta: List[float]):
    zero = np.zeros_like(theta)
    one = np.ones_like(theta)
    refract = np.array(
        [
            [np.cos(theta), zero, np.sin(theta)],
            [zero, one, zero],
            [-np.sin(theta), zero, np.cos(theta)]
        ]
    ).swapaxes(-1, 0)
    return refract


def diagonal(value=1):
    return value * np.identity(3)
