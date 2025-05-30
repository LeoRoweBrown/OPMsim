import warnings
from matplotlib import pyplot as plt
import numpy as np
from .base_element import Element
from .. import matrices


class SineLens(Element):
    """
    Ideal lens that meets Abbe sine condition, realised by a spherical principal surface such that
    the ray height is f1*n1*sin(theta1). Combined with another SineLens with f2, n2, the ray height f2*n2*sin(theta2)
    i.e. the ratio of sines is a constant equal to (f2*n1/f1*n2)

    TODO: Use language with methods like "trace_rays" and "get_optical_transfer_matrix" for electric field.
    trace ray updates the optical matrix tho, so get_optical_transfer_matrix is just a getter?
    """

    def __init__(
            self, NA, focal_length, front_focal_length=None, back_focal_length=None, n=1,
            y_axis_rotation=0, D=None, trace_after=True, update_history=False, flipped_orientation=False):
        self.type = "SineLens"

        # for immersion lenses, f used in the EP = 2*NA*f equation is the back focal length fb = fb*n
        self.focal_length = focal_length
        if front_focal_length is None:
            self.front_focal_length = focal_length * n
        if back_focal_length is None:
            self.back_focal_length = focal_length

        self.NA = NA  # we use effective NA (as if lens were in air)
        self.sine_theta = NA / n
        self.n = n  # object size
        self.y_axis_rotation = y_axis_rotation  # when there is relative tilt between objectives i.e. OPM
        self.flipped_orientation = flipped_orientation

        # used to determine if we stop ray tracing immediately after first surface
        self.trace_after = trace_after
        self.update_history = update_history

        if D is None:  # equation for objective lens
            # D = 2*f_back*NA = 2*f_front*NA/n
            self.D = 2 * self.back_focal_length * NA  # /n
        elif abs(D - 2 * self.back_focal_length * NA) > 1e-6 * D:
            raise ValueError("D, NA and f are not in agreement")
        else:
            self.D = D

    def _normalize(self, v, axis=1):
        norm = np.linalg.norm(v, axis=axis).reshape(v.shape[0], 1, 1)
        norm[norm == 0] = 1
        return v / norm

    def trace_rays(self, rays):
        # If lens is tilted (like O2 and O3), rotate rays relative to objective first
        if self.y_axis_rotation > 0:
            self.rotate_rays_y(rays)

        # Then refract according to lens orientation/whether rays are collimated or not
        if self.flipped_orientation:
            self.focus_collimated_rays(rays)
        else:
            self.collimate_rays(rays)

    def focus_collimated_rays(self, rays):
        """
        Imaging collimated rays to a point such as with the tube lens after an objective

        Args:
            rays (np.ndarray): the N x 3 PolarRays matrix, where N is number of rays
        """
        # First, transform into meridional
        meridional_matrix = matrices.transformation.meridional_transform(rays.phi)

        escape_mask_rho = abs(rays.rho) >= self.focal_length
        rays.escaped = np.logical_or(escape_mask_rho, rays.escaped)
        escape_mask_na = abs(rays.rho) > self.D / 2
        rays.escaped = np.logical_or(escape_mask_na, rays.escaped)

        lens_theta = np.arcsin(rays.rho / self.focal_length)  # positive is anticlockwise rotation
        if any(np.isnan(lens_theta)):
            warnings.warn("NaN values in lens theta - check that rho is not greater than f")

        rays.theta = rays.theta - lens_theta  # consider using k_vec to calculate rather than doing this?
        refract_matrix = matrices.optical_elements.lens_refraction_meridional(-lens_theta)

        rays.rho_before_trace = rays.rho
        if self.trace_after:
            self.trace_f(rays)

        # Then transform back from meridional
        meridional_matrix_inv = matrices.transformation.meridional_transform(rays.phi, inverse=True)

        self.transfer_matrix = meridional_matrix_inv @ refract_matrix @ meridional_matrix \
            @ self.transfer_matrix

    def collimate_rays(self, rays):
        """
        E.g., when rays originate from a point such as with a infinity-corrected primary objective

        Args:
            rays (np.ndarray): the N x 3 PolarRays matrix, where N is number of rays
        """
        # First, transform into meridional
        meridional_matrix = matrices.transformation.meridional_transform(rays.phi)

        self.trace_f(rays)  # trace to first surface

        escape_mask_rho = abs(rays.rho) >= self.focal_length
        rays.escaped = np.logical_or(escape_mask_rho, rays.escaped)
        if any(escape_mask_rho):
            print(np.sum(escape_mask_rho), "rays escaped from rho mask")

        # lens_theta = np.arcsin(rays.rho/self.focal_length)  # positive is anticlockwise rotation
        lens_theta = rays.theta  # ... or just set angles to zero explicitly
        if any(np.isnan(lens_theta)):
            warnings.warn("NaN values in lens theta - check that rho is not greater than f")

        old_theta = rays.theta
        new_theta = rays.theta - lens_theta  # new theta should be zero

        # reject by angle not sine of angle -- avoids wrap-around issue hopefully?
        # TODO remove this, rho rejection should manage this
        escape_mask_na = abs(rays.theta) > np.arcsin(self.sine_theta)
        rays.escaped = np.logical_or(escape_mask_na, rays.escaped)
        if any(escape_mask_na):
            print(np.sum(escape_mask_na), "rays escaped from NA mask")

        rays.theta = new_theta  # assign new theta
        refract_matrix = matrices.optical_elements.lens_refraction_meridional(-lens_theta)

        # EP = 2 * NA * fback = 2 * NA * f_front/n
        # TODO revisit this, I am looking to use front focal length, this scaling might be a hack
        # rays.rho *= self.n  # scale rho for immersion lens

        # TODO small angle approximation - think about doing full arc-based calculation
        # TODO maybe just update rays.area_elements
        rays.area_scaling *= np.abs(np.cos(new_theta) / np.cos(old_theta))

        # Then transform back from meridional
        meridional_matrix_inv = matrices.transformation.meridional_transform(rays.phi, inverse=True)

        self.transfer_matrix = meridional_matrix_inv @ refract_matrix @ meridional_matrix \
            @ self.transfer_matrix

    def trace_f(self, rays):
        """Trace by one focal length"""
        # self.n factor for water immersion objective
        rays.rho = rays.rho + self.front_focal_length * np.sin(rays.theta)
        rays.pos += rays.k_vec * self.front_focal_length  # needs work TODO

    def rotate_rays_y(self, rays):
        """we only rotate our objectives not tube lenses!! this assumes that"""
        y_rot = self.y_axis_rotation
        # y_rot = 0  # for testing

        if rays.isMeridional:
            # print("dot products of ray in merid", np.sum(rays.intensity_vector * rays.k_vec, axis=1))
            rays.meridional_transform(inverse=True)

        # get rotated k_vec
        rotated_k_vec = matrices.transformation.rotate_rays_y(y_rot) @ rays.k_vec
        rays.transfer_matrix = matrices.transformation.rotate_rays_y(y_rot) @ rays.transfer_matrix

        # floating point values sometimes mean k_vec[2] > 1 e.g. 1 + 1e-10
        kz_gt_1_mask = np.squeeze(rotated_k_vec[:, 2] > 1)
        rays.theta[kz_gt_1_mask] = 0
        if rotated_k_vec.shape[0] < 2:  # very annoying edge case when there's one ray and size is 1,3,1
            kz_gt_1_mask = np.reshape(kz_gt_1_mask, (1,))

        rotated_k_vec[kz_gt_1_mask, 2] = 1
        rays.theta = np.arccos(rotated_k_vec[:, 2]).flatten()
        rays.phi = np.arctan2(rotated_k_vec[:, 1], rotated_k_vec[:, 0]).flatten()
        rays.phi = (rays.phi + 2 * np.pi) % (2 * np.pi)

        if np.isnan(rays.phi[np.invert(rays.escaped)]).any():
            raise Exception("NaN(s) in phi after ray rotation in objective")
        if np.isnan(rays.theta[np.invert(rays.escaped)]).any():
            raise Exception("NaN(s) in theta after ray rotation in objective, k = ", rotated_k_vec)

        rays.k_vec = rotated_k_vec
