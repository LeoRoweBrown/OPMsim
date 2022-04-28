from copy import deepcopy
from re import X
import numpy as np
from numpy.ma.core import masked_greater
import optical_matrices
import scipy.interpolate as interp
# Base class for all elements
# I include the possibiliy of off-axis elements and rotated about x even though
# this is unlikely to be supported (maybe rotations will)

class Element():
    def __init__(self):
        self.type = 'Empty element'

    def apply_matrix(self, ray):
        ray.update_history()
        return np.identity(3)


class SineLens(Element):
    """Ideal lens that meets Abbe sine condition"""
    def __init__(self, NA, focal_length, xAxis_rotation = 0, binning_method=False, lens_position=None, direction=1, D=None, isExitPupil=False, endTracingAt=None, dz=0):
        self.type = 'SineLens'
        self.focal_length = focal_length
        self.NA = NA
        self.xAxis_rotation = xAxis_rotation
        # used to determine if we stop ray tracing immediately after first surface
        self.isExitPupil = isExitPupil
        self.endTracingAt = endTracingAt  # for field probing and stuff
        self.dz = dz
        self.lens_position = lens_position
        self.binning_method = binning_method

        if D is None:
            self.D = 2*focal_length*NA
        elif abs(D - 2*focal_length*NA) > 1e-3*D:
            raise ValueError("D, NA and f are not in agreement")
        else:
            self.D = D

    def apply_matrix(self, ray):
        ray.update_history()
        # if self.xAxis_rotation != 0:
            # this is broken and needs fixing - wraparound issue?
        self.rotate_ray_x(ray)
        ray.update_history(("x_axis rotation %.2f rads before lens" % self.xAxis_rotation))
        if not ray.isMeridional:
            ray.E_vec = np.matmul(optical_matrices.meridional_transform(ray.phi), ray.E_vec)
            ray.k_vec = np.matmul(optical_matrices.meridional_transform(ray.phi), ray.k_vec)
            ray.isMeridional = True

        # should just be the same as ray.theta in magnitude because of the constraints
        # theta used in rotaton matrx
        lens_theta = np.arcsin(ray.rho/self.focal_length)  # positive is anticlockwise rotation

        # default case is clockwise rotation, 
        # but this is incorrect for when rays cross optic axis, fixed by using rho but

        old_theta = ray.theta
        new_theta = ray.theta - lens_theta
        # v means first surface is flat (parallel/collimated incoming rays) v
        if abs(old_theta) < abs(new_theta):
            # maybe put the ray checking in here (TODO)
            if abs(ray.rho) > self.D/2:  # Ray escapes system
                ray.E_vec *= 0
            if self.isExitPupil or self.endTracingAt == 1:  # for probing field, tbd
                return #ray
            ray.E_vec = np.matmul(optical_matrices.refraction_meridional(-lens_theta), ray.E_vec)
            ray.k_vec = np.matmul(optical_matrices.refraction_meridional(-lens_theta), ray.k_vec)
            ray.phase += self.focal_length*(1-np.cos(lens_theta))  # extra phase at edges
            # if self.endTracingAt == 2:
            #    return
            ray.theta = new_theta  # assign new theta
            if self.lens_position != 'last':  # if last objective, leave rays along curved pupil
                self.trace_f(ray)

        # we instead add the phase after the second surface and trace first 
        # when the first is curved
        if abs(old_theta) > abs(new_theta):  # means first surface is curved
            if self.isExitPupil:
                raise Warning("Cannot be exit pupil/final element "
                              "- source not imaged (rays are not focussed")
            ray.theta = new_theta  # assign new theta
            if self.lens_position != 'first':  # when rays are generated they're already traced here
                self.trace_f(ray) # trace to first surface
            if abs(ray.rho) > self.D/2:  # Ray escapes system
                ray.E_vec *= 0
            if self.endTracingAt == 1:  # end before refracting rays
                return #ray
            ray.E_vec = np.matmul(optical_matrices.refraction_meridional(-lens_theta), ray.E_vec)
            ray.k_vec = np.matmul(optical_matrices.refraction_meridional(-lens_theta), ray.k_vec)
            ray.phase += self.focal_length*(1-np.cos(lens_theta))  # extra phase at edges
        # cosine scaling, don't scale if binning takes care of ray density change
        if not self.binning_method: 
            ray.E_vec *= (np.cos(new_theta)/np.cos(old_theta))**0.5
        return ray

    def trace_f(self, ray):
        """ Trace by one focal length"""
        ray.rho += self.focal_length*np.sin(ray.theta)
        ray.z += self.focal_length*np.cos(ray.theta)

    def rotate_ray_x(self, ray):
        x = np.sin(ray.theta)*np.cos(ray.phi)
        y = np.sin(ray.theta)*np.sin(ray.phi)
        z = np.cos(ray.theta)

        [x, y, z] = np.matmul(optical_matrices.rotate_rays_x(self.xAxis_rotation), [x, y, z])

        ray.theta = np.arccos(z)
        atan = np.arctan(y/x)  

        atan_p_pi = np.logical_and(x < 0, y >= 0) 
        atan_m_pi = np.logical_and(x < 0, y < 0)
        pi_2 = np.logical_and(x == 0, y > 0)
        m_pi_2 = np.logical_and(x == 0, y < 0) 

        if atan_p_pi:
            ray.phi = atan + np.pi
        elif atan_m_pi:
            ray.phi = atan - np.pi
        elif pi_2:
            ray.phi = np.pi/2
        elif m_pi_2:
            ray.phi = -np.pi/2

        # mod everything, idk
        ray.phi = ray.phi + 2*np.pi % 2*np.pi


class LinearPolariser():
    def __init__(self, psi, dz=0):
        self.type = 'LinearPolariser'
        self.psi = psi
        self.dz = dz

    def apply_matrix(self, ray):
        ray.update_history()
        if ray.isMeridional:  # put back into non meridional basis
            ray.E_vec = np.matmul(\
                optical_matrices.meridional_transform(ray.phi, inverse=True),\
                ray.E_vec)
            ray.k_vec = np.matmul(\
                optical_matrices.meridional_transform(ray.phi, inverse=True),\
                ray.k_vec)
            ray.isMeridional = False
        ray.E_vec = np.matmul(optical_matrices.polariser(self.psi), ray.E_vec)
        ray.k_vec = np.matmul(optical_matrices.polariser(self.psi), ray.E_vec)
        return ray

class WavePlate():
    def __init__(self, psi, delta):
        self.type = 'WavePlate'
        self.psi = psi  # angle of fast axis from x axis
        self.delta = delta  # amount of retardation e.g delta=pi/2 for qwp

    def apply_matrix(self, ray):
        ray.update_history()
        if ray.isMeridional:  # put back into non meridional basis
            ray.E_vec = np.matmul(\
                optical_matrices.meridional_transform(ray.phi, inverse=True),\
                ray.E_vec)
            ray.k_vec = np.matmul(\
                optical_matrices.meridional_transform(ray.phi, inverse=True),\
                ray.k_vec)
        ray.E_vec = np.matmul(optical_matrices.wave_plate(self.psi, self.delta), ray.E_vec)
        ray.k_vec = np.matmul(optical_matrices.wave_plate(self.psi, self.delta), ray.k_vec)
        return ray

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
    def apply_matrix(self, ray):
        ray.update_history()

