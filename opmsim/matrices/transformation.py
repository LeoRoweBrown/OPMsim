"""
matrices representing coordinate transforms during raytracing

"""
import numpy as np
from numpy.typing import ArrayLike, NDArray
from typing import List, Union


def meridional_transform(phi, inverse=False):
    meridional_tensor = np.zeros((len(phi), 3, 3))
    for n in range(len(phi)):
        # First do matrix for a specific value of phi
        meridional_matrix = np.array([
            [np.cos(phi[n]), np.sin(phi[n]), 0],
            [-np.sin(phi[n]), np.cos(phi[n]), 0],
            [0, 0, 1]
        ])
        if inverse:
            try:
                meridional_matrix = np.linalg.inv(meridional_matrix)
            except np.linalg.LinAlgError as e:
                raise Exception(f"Could not invert meridional transform {meridional_matrix}") from e
        # Then assign to the full tensor
        meridional_tensor[n, :, :] = meridional_matrix
    return meridional_tensor

def rotate_y(theta):
    matrix = np.array([
        [np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)]
    ])
    return matrix

def arbitrary_rotation(theta, ux, uy, uz):
    """

    :param theta:
    :param ux:
    :param uy:
    :param uz:
    :return:
    """
    # TODO fix implementation with for loop, the safest way
    raise NotImplementedError
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    rotate = np.array([
        [cos_t + ux ** 2 * (1 - cos_t), ux * uy * (1 - cos_t) - uz * sin_t, ux * uz * (1 - cos_t) + uy * sin_t],
        [uy * ux * (1 - cos_t) + uz * sin_t, cos_t + uy ** 2 * (1 - cos_t), uy * uz * (1 - cos_t) - ux * sin_t],
        [uz * ux * (1 - cos_t) - uy * sin_t, uz * uy * (1 - cos_t) + ux * sin_t, cos_t + uz ** 2 * (1 - cos_t)]
    ])
    if hasattr(ux, "__len__") and len(ux) > 1:
        rotate = np.swapaxes(rotate, 0, 2)
        # rotate = rotate.reshape((rotate.shape[2], 3, 3))
    return rotate


def local_wavefront_to_lab_basis(phi: ArrayLike, theta: ArrayLike) -> np.ndarray:
    """
    Inverse version of lab_to_local_wavefront_basis.

    :param float | ndarray phi: azimuthal angle of ray(s)
    :param float | ndarray theta: polar angle of rays(s)
    :param bool inverse: if True, transforms local wavefront coordinates (Iz=0) to lab coordinates
    :return: transformation matrix
    :rtype np.ndarray
    """
    return lab_to_local_wavefront_basis(phi, theta, inverse=True)

def lab_to_local_wavefront_basis(phi: ArrayLike, theta: ArrayLike, inverse=False) -> np.ndarray:
    """
    Used to calculate electrics fields evaluated on a curved wavefront, e.g. in detection stage
    changes coordinate system according to k vector so that Iz = 0
    both rotates into meridional and then does theta rotation

    Args:
        phi (ArrayLike): azimuthal angle of ray(s)
        theta (ArrayLike): polar angle of rays(s)
        inverse (bool, optional): if True, transforms local wavefront coordinates (Iz=0) to lab coordinates.
            Defaults to False.

    Raises:
        Exception: if phi and theta are not equal in length
        Exception: if matrix inversion fails (np.linalg.LinAlgError)

    Returns:
        np.ndarray: tensor for rotating into a basis local to (phi, theta) points so Iz = 0
    """
    phi = np.atleast_1d(phi)
    theta = np.atleast_1d(theta)
    if len(phi) != len(theta):
        raise Exception(f"length of phi and theta arrays not equal ({len(phi)} and {len(theta)})")
    rotate_tensor = np.zeros((len(phi), 3, 3))
    for n in range(len(phi)):
        rotate_matrix = np.array([
            [np.cos(phi) * np.cos(theta), np.sin(phi) * np.cos(theta), -np.sin(theta)],
            [-np.sin(phi), np.cos(phi), np.zeros_like(phi)],
            [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)]
        ]).reshape((len(phi), 3, 3))
        if inverse:
            try:
                rotate_matrix = np.linalg.inv(rotate_matrix)
            except np.linalg.LinAlgError as e:
                raise Exception("Failed to invert local lab rotation matrix") from e
        rotate_tensor[n, :, :] = rotate_matrix
    return rotate_tensor

def ps_projection_matrix(p, s, k_vec, inverse=False):
    """project E onto p and s component in reflection, p = k1 x N, s = k1 x p"""
    # for loop instead of vectorise just for safety.. reshape likes to muddle up elements
    ps_projection = np.zeros((p.shape[0], 3, 3))

    if k_vec.ndim < 2:  # single k-vec -- single ray
        k_vec = np.expand_dims(k_vec, 0)
    if p.ndim < 2:  # single p-vec -- single ray
        p = np.expand_dims(p, 1)

    for n in range((p.shape[0])):

        if np.all(np.equal(p[n, :], 0)):
            ps_projection[n, :, :] = np.identity(3)
        else:
            ps_projection[n, :, :] = np.array(
                [
                    [p[n, 0], p[n, 1], p[n, 2]],
                    [s[n, 0], s[n, 1], s[n, 2]],
                    [k_vec[n, 0], k_vec[n, 1], k_vec[n, 2]]
                ]
            )
        if inverse:
            ps_projection[n, :, :] = np.linalg.inv(ps_projection[n, :, :])
    return ps_projection

def flip_axis(axis=2):
    flip = np.identity(3)
    flip[axis, axis] = -1
    return flip

def rotate_rays_x(angle: float):
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

def rotate_rays_y(angle: float):
    rot_mat_y = np.array([
        [np.cos(angle), 0, np.sin(angle)],
        [0, 1, 0],
        [-np.sin(angle), 0, np.cos(angle)]
    ])
    return rot_mat_y

def reflection_cartesian_matrix(normal_vec: Union[List[float], NDArray]):
    # householder transformation, may need to replace - with +

    reflect_matrix = [
        [1 - 2 * normal_vec[0] ** 2, -2 * normal_vec[0] * normal_vec[1], -2 * normal_vec[0] * normal_vec[2]],
        [-2 * normal_vec[0] * normal_vec[1], 1 - 2 * normal_vec[1] ** 2, -2 * normal_vec[1] * normal_vec[2]],
        [-2 * normal_vec[0] * normal_vec[2], -2 * normal_vec[1] * normal_vec[2], 1 - 2 * normal_vec[2] ** 2]
    ]
    return reflect_matrix