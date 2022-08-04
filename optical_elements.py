from cmath import isnan, nan
from copy import deepcopy
from logging import warning
from re import X
import numpy as np
from numpy.ma.core import masked_greater
import optical_matrices
import scipy.interpolate as interp
from matplotlib import pyplot as plt
import graphics

# Base class for all elements
# I include the possibiliy of off-axis elements and rotated about x even though
# this is unlikely to be supported (maybe rotations will)

class Element():
    def __init__(self):
        self.type = 'Empty element'

    def apply_matrix(self, rays):
        rays.update_history()
        return np.identity(3)

class SineLens(Element):
    """Ideal lens that meets Abbe sine condition"""
    def __init__(self, NA, focal_length, xAxis_rotation = 0, binning_method=False,
                n=1, D=None):
        self.type = 'SineLens'
        self.focal_length = focal_length
        self.NA = NA/n  # we use effective NA (as if lens were in air)
        self.xAxis_rotation = xAxis_rotation
        # used to determine if we stop ray tracing immediately after first surface

        self.binning_method = binning_method

        if D is None:
            self.D = 2*focal_length*NA
        elif abs(D - 2*focal_length*NA) > 1e-6*D:
            raise ValueError("D, NA and f are not in agreement")
        else:
            self.D = D

    def apply_matrix(self, rays):
        rays.update_history()

        if any(np.isnan(rays.phi[np.invert(rays.escaped)])):
            raise Exception("NaN(s) in phi after ray rotation in objective")
        if self.xAxis_rotation != 0:
            print("rotating rays by", self.xAxis_rotation)
            self.rotate_ray_x(rays)  # for angled lenses i.e. O3
            rays.update_history(("x_axis rotation %.2f rads before lens" % self.xAxis_rotation))
        if not rays.isMeridional:
            # I don't trust this, need to fix it? use multiplication here for now
            # horrible loop method to check:
            # for n in range(len(rays.I_vec)):
            #     rays.I_vec[n,:,:] = \
            #         optical_matrices.meridional_transform(rays.phi[n]) @ rays.I_vec[n,:,:]
            #     rays.k_vec[n,:,:] = \
            #         optical_matrices.meridional_transform(rays.phi[n]) @ rays.k_vec[n,:,:]

            # rays.I_vec = optical_matrices.meridional_transform_tensor(rays.phi) @ rays.I_vec
            # rays.k_vec = optical_matrices.meridional_transform_tensor(rays.phi) @ rays.k_vec
            # rays.isMeridional = True
            rays.meridional_transform()

        # default case is clockwise rotation, 
        # but this is incorrect for when rays cross optic axis, fixed by using rho but
        first_curved = False
        old_theta = rays.theta

        print("minumum of intensities (x)", np.min(rays.I_vec[:,0]))
        print("minumum of intensities (y)", np.min(rays.I_vec[:,1]))

        x = rays.rho*np.cos(rays.phi)
        y = rays.rho*np.sin(rays.phi)

        if all(rays.rho == 0):
            x = np.sin(rays.theta)*np.cos(rays.phi)
            y = np.sin(rays.theta)*np.sin(rays.phi) 

        merid_pupil = graphics.PupilPlotObject(x, y, rays.I_vec[:,0],  rays.I_vec[:,1])
        merid_pupil.plot("meridian pupil during lens")

        # this condition means first surface is flat (parallel/collimated incoming rays)
        if all(abs(old_theta[np.invert(rays.escaped)] < 1e-10)):  # soft inequality for theta = 0
            first_curved = False
            
            escape_mask_rho = abs(rays.rho) >= self.focal_length
            rays.escaped = np.logical_or(escape_mask_rho, rays.escaped)

            lens_theta = np.arcsin(rays.rho/self.focal_length)  # positive is anticlockwise rotation
            new_theta = rays.theta - lens_theta

            # if abs(rays.rho) > self.D/2:  # Ray escapes system
            escape_mask_na = abs(rays.theta) > np.arcsin(self.NA)
            rays.escaped = np.logical_or(escape_mask_na, rays.escaped)

            rays.theta = new_theta  # consider using k_vec to calculate rather than doing this?

            rays.I_vec = \
                optical_matrices.refraction_meridional_tensor(-lens_theta) @ rays.I_vec
            rays.k_vec = \
                optical_matrices.refraction_meridional_tensor(-lens_theta) @ rays.k_vec

            self.trace_f(rays)

        # when the first is curved
        
        else:  # means first surface is curved
            first_curved = True
            self.trace_f(rays) # trace to first surface

            escape_mask_rho = abs(rays.rho) >= self.focal_length
            rays.escaped = np.logical_or(escape_mask_rho, rays.escaped)

            lens_theta = np.arcsin(rays.rho/self.focal_length)  # positive is anticlockwise rotation
            new_theta = rays.theta - lens_theta
            lens_theta[np.abs(lens_theta) < 1e-9] = 0  # avoid negative values from floating point error

            plt.figure()
            plt.hist(lens_theta, bins=20)
            plt.title("new theta")
            plt.show()
            plt.figure()
            plt.hist(rays.phi)
            plt.title("phi")
            plt.show()

            # if abs(rays.rho) > self.D/2:  # Ray escapes system
            escape_mask_na = abs(rays.theta) > np.arcsin(self.NA)
            rays.escaped = np.logical_or(escape_mask_na, rays.escaped)
            print("na escape mask")
            print(escape_mask_na)

            rays.theta = new_theta  # assign new theta

            rays.I_vec = \
                optical_matrices.refraction_meridional_tensor(-lens_theta) @ rays.I_vec
            rays.k_vec = \
                optical_matrices.refraction_meridional_tensor(-lens_theta) @ rays.k_vec

            rays.meridional_transform(inverse=True)
            refract_pupil = graphics.PupilPlotObject(x, y, rays.I_vec[:,0],  rays.I_vec[:,1])
            refract_pupil.plot("pupil after refraction")

            print("minumum of intensities after refraction", np.min(rays.I_vec))


        # small angle approximation - think about doing full arc-based calculation
        #TODO: check this *= change
        print("min old theta", np.min(old_theta))
        print("min new theta", np.min(new_theta))
        # absolute for good mesure
        rays.area_scaling *= np.abs(np.cos(new_theta)/np.cos(old_theta))
        print("scaling", rays.area_scaling)
        print("min of scaling", np.min(rays.area_scaling))

        # check for NaNs (but ignore escaped rays)
        if np.isnan(rays.phi[np.invert(rays.escaped)]).any():
            raise Exception("NaN(s) in phi after objective")
        # I_vec has more dims, so need .any()
        if np.isnan(rays.I_vec[np.invert(rays.escaped)]).any():
            raise Exception("NaN(s) in I-vector after objective")

        # check dot productive of z and yx - should be zero
        I_k_dot = np.sum(rays.I_vec * rays.k_vec, axis=1)
        # print(I_k_dot)
        return rays

    def trace_f(self, rays):
        """ Trace by one focal length"""
        rays.rho = rays.rho + self.focal_length*np.sin(rays.theta)
        # rays.z += self.focal_length*np.cos(rays.theta)

    def rotate_ray_x(self, rays):
        """we only rotate our objectives not tube lenses!! this assumes that"""

        if rays.isMeridional:
            # print("dot products of ray in merid", np.sum(rays.I_vec * rays.k_vec, axis=1))
            rays.meridional_transform(inverse=True)

        rays.k_vec = optical_matrices.rotate_rays_x(self.xAxis_rotation) @ rays.k_vec
        rays.I_vec = optical_matrices.rotate_rays_x(self.xAxis_rotation) @ rays.I_vec

        # floating point values sometimes mean k_vec[2] > 1 e.g. 1 + 1e-10        
        kz_gt_1_mask = np.squeeze(rays.k_vec[:,2] > 1)
        rays.theta[kz_gt_1_mask] = 0
        rays.phi = np.arctan2(rays.k_vec[:,1], rays.k_vec[:,0])

        if np.isnan(rays.phi[np.invert(rays.escaped)]).any():
            raise Exception("NaN(s) in phi after ray rotation in objective")
        if np.isnan(rays.theta[np.invert(rays.escaped)]).any():
            raise Exception("NaN(s) in theta after ray rotation in objective, k = ", rays.k_vec)

        # recompute rho
        rays.rho = self.focal_length*np.sin(rays.theta)

class LinearPolariser():
    def __init__(self, psi, dz=0):
        self.type = 'LinearPolariser'
        self.psi = psi
        self.dz = dz

    def apply_matrix(self, rays):
        rays.update_history()
        if rays.isMeridional:  # put back into non meridional basis
            rays.meridional_transform(inverse=True)
        rays.I_vec = optical_matrices.polariser(self.psi) @ rays.I_vec
        rays.k_vec = optical_matrices.polariser(self.psi) @ rays.k_vec
        print("dot product after polariser", np.sum(rays.I_vec * rays.k_vec, axis=1))
        return rays

class WavePlate():
    def __init__(self, psi, delta):
        self.type = 'WavePlate'
        self.psi = psi  # angle of fast axis from x axis
        self.delta = delta  # amount of retardation e.g delta=pi/2 for qwp

    def apply_matrix(self, rays):
        rays.update_history()
        if rays.isMeridional:  # put back into non meridional basis
            rays.meridional_transform(inverse=True)
        rays.I_vec = optical_matrices.wave_plate(self.psi, self.delta) @ rays.I_vec
        rays.k_vec = optical_matrices.wave_plate(self.psi, self.delta) @ rays.k_vec
        return rays

class IdealFlatMirrorNoRotation():
    pass

class IdealFlatMirror():
    """
    Ideal flat mirror with rotation about x axis
    May try making mirror with different p and s reflectance but requires
    decomposition of these components and it's within 1% difference anyway.
    """
    def __init__(self, focal_length, diameter, dz=0, rot_x=0):
        self.type = 'IdealFlatMirror'
        self.reflectance = 1
        self.diameter = diameter
        self.theta = 0
        self.phi = 0
        ## TODO work out how to find intersection and angle between mirror normal and ray
        # maybe change coord system then back as soon as reflection is done?
    def apply_matrix(self, rays):
        rays.update_history()

