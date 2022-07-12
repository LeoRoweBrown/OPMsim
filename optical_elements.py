from cmath import isnan, nan
from copy import deepcopy
from logging import warning
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
        
        print("theta at START of apply matrix of lens:", ray.theta, "rho:", ray.rho, "escaped? ", ray.escaped,\
            "rot", self.xAxis_rotation)
        # print("applying matrix see if phi is nan")
        if np.isnan(ray.phi):
            raise Exception("NaN in phi after ray rotation in objective")
        if self.xAxis_rotation != 0:
            # this is broken and needs fixing - wraparound issue?
            print("rotating rays by", self.xAxis_rotation)
            self.rotate_ray_x(ray)  # for angled lenses i.e. O3
            ray.update_history(("x_axis rotation %.2f rads before lens" % self.xAxis_rotation))
        if not ray.isMeridional:
            ray.E_vec = np.matmul(optical_matrices.meridional_transform(ray.phi), ray.E_vec)
            ray.k_vec = np.matmul(optical_matrices.meridional_transform(ray.phi), ray.k_vec)
            ray.isMeridional = True

        # should just be the same as ray.theta in magnitude because of the constraints
        # theta used in rotaton matrx
        # print("ARCSINE ARG", (ray.rho/self.focal_length))
        # if abs(ray.rho) >= self.focal_length:
        #     ray.escaped = True  # not physical for sine lens to be able to focus this
        #     lens_theta = ray.rho/abs(ray.rho)*np.pi/2  # set magnitude to 90 degrees
        # else:
        #     lens_theta = np.arcsin(ray.rho/self.focal_length)  # positive is anticlockwise rotation

        # default case is clockwise rotation, 
        # but this is incorrect for when rays cross optic axis, fixed by using rho but
        first_curved = False
        old_theta = ray.theta
        # v means first surface is flat (parallel/collimated incoming rays) v
        if abs(old_theta) < 1e-10:  # soft inequality for theta = 0
            first_curved = False
            # 2022 07 01 - put this in if statements to get order right:
            print("ARCSINE ARG", (ray.rho/self.focal_length))
            if abs(ray.rho) >= self.focal_length:
                ray.escaped = True  # not physical for sine lens to be able to focus this
                lens_theta = ray.rho/abs(ray.rho)*np.pi/2  # set magnitude to 90 degrees
            else:
                lens_theta = np.arcsin(ray.rho/self.focal_length)  # positive is anticlockwise rotation
            new_theta = ray.theta - lens_theta
            ## -----------------------------
            if abs(ray.rho) > self.D/2:  # Ray escapes system
                print("ray escaped!")
                # raise Exception("ray escaped")
                ray.E_vec *= 0
                ray.theta = 0  # for safety, consider changing (TODO)
                ray.phi = 0
                ray.rho = 0
                ray.escaped = True  # return a Nonetype?
            else:
                ray.theta = new_theta  # consider using k_vec to calculate rather than doing this?
            # v remove this TODO v
            if self.isExitPupil or self.endTracingAt == 1:  # for probing field, tbd, remove later?
                return #ray
            ray.E_vec = np.matmul(optical_matrices.refraction_meridional(-lens_theta), ray.E_vec)
            ray.k_vec = np.matmul(optical_matrices.refraction_meridional(-lens_theta), ray.k_vec)
            ray.phase += self.focal_length*(1-np.cos(lens_theta))  # extra phase at edges
            # if self.endTracingAt == 2:
            #    return
            # if self.lens_position != 'last':  # if last objective, leave rays along curved pupil
            self.trace_f(ray)

        # we instead add the phase after the second surface and trace first 
        # when the first is curved
        if abs(old_theta) > 1e-10:  # means first surface is curved
            first_curved = True
            if self.isExitPupil:
                raise Warning("Cannot be exit pupil/final element "
                              "- source not imaged (rays are not focussed")
            print("ray before tracing to curve, rho:", ray.rho)
            if self.lens_position != 'first':  # when rays are generated they're already traced here
                self.trace_f(ray) # trace to first surface
            print("after tracing ray, rho:", ray.rho)
            # 2022 07 01 - put this in if statements to get order right:
            print("ARCSINE ARG", (ray.rho/self.focal_length))
            if abs(ray.rho) >= self.focal_length:
                ray.escaped = True  # not physical for sine lens to be able to focus this
                lens_theta = ray.rho/abs(ray.rho)*np.pi/2  # set magnitude to 90 degrees
            else:
                lens_theta = np.arcsin(ray.rho/self.focal_length)  # positive is anticlockwise rotation
            new_theta = ray.theta - lens_theta
            ## -----------------------------


            # if abs(ray.rho) > self.D/2:  # Ray escapes system
            if abs(ray.theta) > np.arcsin(self.NA):
                print("ray escaped!")
                # raise Exception("ray escaped")
                ray.E_vec *= 0
                ray.theta = 0  # for safety
                ray.phi = 0
                ray.rho = 0
                ray.escaped = True
            else:
                ray.theta = new_theta  # assign new theta

            if self.endTracingAt == 1:  # end before refracting rays
                return #ray
            ray.E_vec = np.matmul(optical_matrices.refraction_meridional(-lens_theta), ray.E_vec)
            ray.k_vec = np.matmul(optical_matrices.refraction_meridional(-lens_theta), ray.k_vec)
            ray.phase += self.focal_length*(1-np.cos(lens_theta))  # extra phase at edges
        # cosine scaling, don't scale if binning takes care of ray density change
        ray.area_scaling = (np.cos(new_theta)/np.cos(old_theta))**0.5
        if not self.binning_method: 
            ray.E_vec *= (np.cos(new_theta)/np.cos(old_theta))**0.5

        print("theta at end of apply matrix of lens:", ray.theta, "rho:", ray.rho, "escaped? ", ray.escaped, "new theta:", new_theta, "lens theta", lens_theta,\
            "first surf curved", first_curved, "rot", self.xAxis_rotation)
        # print("finished applying matrix see if phi is nan again")
        if np.isnan(ray.phi):
            raise Exception("NaN in phi after objective")
        if any(np.isnan(ray.E_vec)) and not ray.escaped:
            raise Exception("NaN in E-vector after objective")
        return ray

    def trace_f(self, ray):
        """ Trace by one focal length"""
        ray.rho += self.focal_length*np.sin(ray.theta)
        ray.z += self.focal_length*np.cos(ray.theta)

    def rotate_ray_x(self, ray):
        """we only rotate our objectives not tube lenses!! this assumes that"""
        # x = np.sin(ray.theta)*np.cos(ray.phi)
        # y = np.sin(ray.theta)*np.sin(ray.phi)
        # z = np.cos(ray.theta)
        # k_vec_before = deepcopy(ray.k_vec)
        # first make sure not under meridional translation
        
        # make all negative thetas into positive thetas with a phi + 180 rotation
        # if ray.theta < 0:
        #     ray.theta = np.abs(ray.theta)
        #    ray.phi = (ray.phi + np.pi) % (2*np.pi)

        if ray.isMeridional:
            print("dot products of ray in merid", np.dot(ray.E_vec, ray.k_vec))
            ray.E_vec = np.matmul(optical_matrices.meridional_transform(ray.phi, inverse=True), ray.E_vec)
            ray.k_vec = np.matmul(optical_matrices.meridional_transform(ray.phi, inverse=True), ray.k_vec)
            print("dot products of ray after inverse merid", np.dot(ray.E_vec, ray.k_vec))

        ray.k_vec = np.matmul(optical_matrices.rotate_rays_x(self.xAxis_rotation), ray.k_vec)
        ray.E_vec = np.matmul(optical_matrices.rotate_rays_x(self.xAxis_rotation), ray.E_vec)
        # print(ray.k_vec - k_vec_before) 
        # floating point values sometimes mean k_vec[2] > 1 e.g. 1 + 1e-10
        printed_before = False
        if abs(ray.theta) > 1e-6:
            # print("theta before", ray.theta, flush=True)
            printed_before = True
        # ray.theta = np.arccos(ray.k_vec[2]/np.sqrt(ray.k_vec.dot(ray.k_vec)))
        
        if abs(ray.k_vec[2]) > 1:
            ray.theta = 0
            raise ValueError("Kz > 1")
            print("warning: kz > 1")
        else:
            # ray.theta = np.arccos(ray.k_vec[2])
            mag = (np.sum(ray.k_vec*ray.k_vec))**0.5
            cos_theta = ray.k_vec[2]/mag
            theta_before = deepcopy(ray.theta)
            ray.theta = np.arccos(cos_theta)
            print("mag", mag, "cos", cos_theta, "theta after rot", ray.theta, "theta before", theta_before)
            # print("cos", cos_theta)
        if printed_before:
            pass # print("theta after rot", ray.theta, flush=True)
        # atan = np.arctan(ray.k_vec[1]/ray.k_vec[0])  # phi
        ray.phi = np.arctan2(ray.k_vec[1], ray.k_vec[0])
        # atan_p_pi = np.logical_and(ray.k_vec[0] < 0, ray.k_vec[1] >= 0) 
        # atan_m_pi = np.logical_and(ray.k_vec[0] < 0, ray.k_vec[1] < 0)
        # pi_2 = np.logical_and(ray.k_vec[0] == 0, ray.k_vec[1] > 0)
        # m_pi_2 = np.logical_and(ray.k_vec[0] == 0, ray.k_vec[1] < 0)
        # phi0 = np.logical_and(ray.k_vec[0] == 0, ray.k_vec[1] == 0)

        # if atan_p_pi:
        #     ray.phi = atan + np.pi
        # elif atan_m_pi:
        #     ray.phi = atan - np.pi
        # elif pi_2:
        #     ray.phi = np.pi/2
        # elif m_pi_2:
        #     ray.phi = 2*np.pi-np.pi/2
        # elif phi0:
        #     print("ray phi print")
        #     ray.phi = 0
        # else:
        #     ray.phi = atan

        # mod everything, idk
        # ray.phi = (ray.phi + 2*np.pi) % (2*np.pi)

        if np.isnan(ray.phi):
            raise Exception("NaN in phi after ray rotation in objective")
        if np.isnan(ray.theta):
            raise Exception("NaN in theta after ray rotation in objective, k = ", ray.k_vec)

        # recompute rho
        ray.rho = self.focal_length*np.sin(ray.theta)



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

