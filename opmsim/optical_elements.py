import warnings
from copy import deepcopy
import numpy as np
import scipy.interpolate as interp
from matplotlib import pyplot as plt

from .tools import graphics
from . import optical_matrices

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
    def __init__(self, NA, focal_length, n=1, yAxis_rotation = 0, binning_method=False,
            D=None, show_plots=False):
        self.type = 'SineLens'
        self.focal_length = focal_length
        self.NA = NA  # we use effective NA (as if lens were in air)
        self.sine_theta = NA/n
        self.n = n
        self.yAxis_rotation = yAxis_rotation
        # used to determine if we stop ray tracing immediately after first surface
        self.binning_method = binning_method
        self.show_plots = show_plots

        if D is None:
            self.D = 2*focal_length*NA*n
        elif abs(D - 2*focal_length*NA*n) > 1e-6*D:
            raise ValueError("D, NA and f are not in agreement")
        else:
            self.D = D

    def apply_matrix(self, rays, update_history=False):
        if update_history: rays.update_history()

        if any(np.isnan(rays.phi[np.invert(rays.escaped)])):
            raise Exception("NaN(s) in phi after ray rotation in objective")
        if self.yAxis_rotation != 0:
            print("rotating rays by", self.yAxis_rotation)
            # print("before rotating theta", rays.theta)
            # print("before rotating rho" , rays.rho)
            self.rotate_ray_y(rays)  # for angled lenses i.e. O3
            if update_history: rays.update_history(("x_axis rotation %.2f rads before lens" % self.yAxis_rotation))
        if not rays.isMeridional:
            rays.meridional_transform()

        # default case is clockwise rotation, 
        # but this is incorrect for when rays cross optic axis, fixed by using rho but
        old_theta = rays.theta

        if all(rays.rho == 0):
            x = np.sin(rays.theta)*np.cos(rays.phi)
            y = np.sin(rays.theta)*np.sin(rays.phi) 

        # this condition means first surface is flat (parallel/collimated incoming rays)
        if all(abs(old_theta[np.invert(rays.escaped)]) < 1e-10):  # soft inequality for theta = 0
            print("FLAT REFRACTION")

            # print(rays.rho)
            # print(self.n)
            rays.rho = rays.rho/self.n  # scale rho for immersion lens

            escape_mask_rho = abs(rays.rho) >= self.focal_length
            rays.escaped = np.logical_or(escape_mask_rho, rays.escaped)

            lens_theta = np.arcsin(rays.rho/self.focal_length)  # positive is anticlockwise rotation
            if any(np.isnan(lens_theta)):
                warnings.warn("NaN values in lens theta - check that rho is not greater than f")
                print("rho:", rays.rho)
                print("f:", self.focal_length)
                print("rho/f:", rays.rho/self.focal_length)

            new_theta = rays.theta - lens_theta

            # if abs(rays.rho) > self.D/2:  # Ray escapes system
            escape_mask_na = abs(rays.rho) > self.D/2
            # escape_mask_na = abs(rays.theta) > np.arcsin(self.NA)
            rays.escaped = np.logical_or(escape_mask_na, rays.escaped)

            rays.theta = new_theta  # consider using k_vec to calculate rather than doing this?

            rays.k_vec = \
                optical_matrices.refraction_meridional_tensor(-lens_theta) @ rays.k_vec
            rays.transfer_matrix = \
                optical_matrices.refraction_meridional_tensor(-lens_theta) @ rays.transfer_matrix

            self.trace_f(rays)

            # plot x y 
            x = np.sin(rays.theta)*np.cos(rays.phi)
            y = np.sin(rays.theta)*np.sin(rays.phi)

            phi_ring = np.linspace(0,2*np.pi,100)
            x_NA_ring = self.sine_theta * np.cos(phi_ring)
            y_NA_ring = self.sine_theta * np.sin(phi_ring)

            if self.show_plots:
                plt.figure()
                plt.scatter(x, y)
                plt.scatter(x_NA_ring, y_NA_ring, s=3, c='blue')
                plt.gca().axis('equal')
                plt.show()
                rays.quiver_plot(self.show_plots)

        # when the first is curved
        else:  # means first surface is curved
            print("CURVED REFRACTION")
            self.trace_f(rays) # trace to first surface

            # plot x y 
            x = np.sin(rays.theta)*np.cos(rays.phi)
            y = np.sin(rays.theta)*np.sin(rays.phi)

            phi_ring = np.linspace(0,2*np.pi,100)
            x_NA_ring = self.sine_theta * np.cos(phi_ring)
            y_NA_ring = self.sine_theta * np.sin(phi_ring)
        
            if self.show_plots:
                plt.figure()
                plt.scatter(x, y)
                plt.scatter(x_NA_ring, y_NA_ring, s=3, c='blue')
                plt.gca().axis('equal')
                plt.title("Before refraction at curved surface")
                plt.show()

                # TODO comment this out
                rays.quiver_plot(self.show_plots)

            escape_mask_rho = abs(rays.rho) >= self.focal_length
            rays.escaped = np.logical_or(escape_mask_rho, rays.escaped)
            if any(escape_mask_rho):
                print(np.sum(escape_mask_rho), "escaped from rho mask")

            lens_theta = np.arcsin(rays.rho/self.focal_length)  # positive is anticlockwise rotation
            if any(np.isnan(lens_theta)):
                warnings.warn("NaN values in lens theta - check that rho is not greater than f")
                # print("rho:", rays.rho)
                # print("f:", self.focal_length)
                # print("rho/f:", rays.rho/self.focal_length)

            new_theta = rays.theta - lens_theta
            
            lens_theta[np.abs(lens_theta) < 1e-9] = 0  # avoid negative values from floating point error

            # if abs(rays.rho) > self.D/2:  # Ray escapes system
            escape_mask_na = abs(rays.theta) > np.arcsin(self.sine_theta)
            rays.escaped = np.logical_or(escape_mask_na, rays.escaped)
            if any(escape_mask_na):
                print(np.sum(escape_mask_na), "escaped from NA mask")
                # print("sine_theta", self.sine_theta)
                # print("max sin(theta)", np.max(np.sin(np.abs(rays.theta))))

            rays.theta = new_theta  # assign new theta

            rays.k_vec = \
                optical_matrices.refraction_meridional_tensor(-lens_theta) @ rays.k_vec
            rays.transfer_matrix = \
                optical_matrices.refraction_meridional_tensor(-lens_theta) @ rays.transfer_matrix

            rays.meridional_transform(inverse=True)
            rays.rho *= self.n  # scale rho for immersion lens


        # small angle approximation - think about doing full arc-based calculation
        rays.area_scaling *= np.abs(np.cos(new_theta)/np.cos(old_theta))

        if self.show_plots and (np.abs(rays.rho)>1e-9).any():  # show at the end 
            print("plotting with forced rho")
            print(np.max(rays.rho))
            rays.quiver_plot(use_rho=True)

        # check for NaNs (but ignore escaped rays)
        if np.isnan(rays.phi[np.invert(rays.escaped)]).any():
            raise Exception("NaN(s) in phi after objective")

        return rays

    def trace_f(self, rays):
        """ Trace by one focal length"""
        # self.n factor for water immersion objective 
        rays.rho = rays.rho + self.focal_length*np.sin(rays.theta)
        # rays.z += self.focal_length*np.cos(rays.theta)

    def rotate_ray_y(self, rays):
        """we only rotate our objectives not tube lenses!! this assumes that"""
        y_rot = self.yAxis_rotation
        # y_rot = 0  # for testing

        if rays.isMeridional:
            # print("dot products of ray in merid", np.sum(rays.I_vec * rays.k_vec, axis=1))
            rays.meridional_transform(inverse=True)

        # print("k before", rays.k_vec[0:5,:])
        # print("phi and theta before", rays.phi[0:5], rays.theta[0:5])
        # print("max theta before", np.max(rays.theta))
        # print("min theta before", rays.theta.min())
        # print("min kz before", rays.k_vec[:,2].min())

        # rotate E (and k) into local basis
        # rays.rotate_rays_local_basis(inverse_meridional=True)
        kx = np.sin(rays.theta)*np.cos(rays.phi)
        ky = np.sin(rays.theta)*np.sin(rays.phi)
        kz = np.cos(rays.theta)

        if self.show_plots:
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            plt.scatter(kx,ky,kz)
            plt.title("points before")
            ax = fig.add_subplot()
            plt.scatter(kx,kz)
            plt.xlim([-1.2,1.2])
            plt.ylim([-1.2,1.2])
            plt.gca().axis('equal')
            plt.show()

        # get rotated k_vec
        rotated_k_vec = optical_matrices.rotate_rays_y(y_rot) @ rays.k_vec
        rays.transfer_matrix = optical_matrices.rotate_rays_y(y_rot) @ rays.transfer_matrix

        # rays.meridional_transform()
        # rays.rotate_rays_local_basis(inverse_basis=True)

        # floating point values sometimes mean k_vec[2] > 1 e.g. 1 + 1e-10        
        kz_gt_1_mask = np.squeeze(rotated_k_vec[:,2] > 1)
        rays.theta[kz_gt_1_mask] = 0
        # print(kz_gt_1_mask)
        # print(rotated_k_vec.shape)
        # print(rotated_k_vec)
        if rotated_k_vec.shape[0] < 2:  # very annoying edge case when size is 1,3,1 still don't really understand it
            #if kz_gt_1_mask:
            kz_gt_1_mask = np.reshape(kz_gt_1_mask, (1,))
        rotated_k_vec[kz_gt_1_mask, 2] = 1
        rays.theta = np.arccos(rotated_k_vec[:,2]).flatten()
        rays.phi = np.arctan2(rotated_k_vec[:,1], rotated_k_vec[:,0]).flatten()
        rays.phi = (rays.phi + 2*np.pi) % (2*np.pi)

        if np.isnan(rays.phi[np.invert(rays.escaped)]).any():
            raise Exception("NaN(s) in phi after ray rotation in objective")
        if np.isnan(rays.theta[np.invert(rays.escaped)]).any():
            raise Exception("NaN(s) in theta after ray rotation in objective, k = ", rotated_k_vec)

        rays.k_vec = rotated_k_vec

        kx = np.sin(rays.theta)*np.cos(rays.phi)
        ky = np.sin(rays.theta)*np.sin(rays.phi)
        kz = np.cos(rays.theta)
        
        if self.show_plots:
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            plt.scatter(kx,ky,kz)
            plt.title("points after")
            ax = fig.add_subplot()
            plt.scatter(kx,kz)
            plt.xlim([-1.2,1.2])
            plt.ylim([-1.2,1.2])
            plt.gca().axis('equal')
            plt.show()

class LinearPolariser():
    def __init__(self, psi, dz=0):
        self.type = 'LinearPolariser'
        self.psi = psi
        self.dz = dz

    def apply_matrix(self, rays, update_history=False):
        if update_history: rays.update_history()
        if rays.isMeridional:  # put back into non meridional basis
            rays.meridional_transform(inverse=True)
        # rays.k_vec = optical_matrices.polariser(self.psi) @ rays.k_vec
        rays.transfer_matrix = optical_matrices.polariser(self.psi) @ rays.transfer_matrix
        return rays

class WavePlate():
    def __init__(self, psi, delta):
        self.type = 'WavePlate'
        self.psi = psi  # angle of fast axis from x axis
        self.delta = delta  # amount of retardation e.g delta=pi/2 for qwp

    def apply_matrix(self, rays, update_history=False):
        if update_history: rays.update_history()
        if rays.isMeridional:  # put back into non meridional basis
            rays.meridional_transform(inverse=True)
        # rays.k_vec = optical_matrices.wave_plate(self.psi, self.delta) @ rays.k_vec
        rays.transfer_matrix = \
            optical_matrices.wave_plate(self.psi, self.delta) @ rays.transfer_matrix
        return rays

class IdealFlatMirrorNoRotation():
    pass

class FlatMirror():
    """
    Flat mirror with rotation about y axis
    """
    def __init__(self, rot_y, film_thickness=100e-6, 
        n_film_file='../refractive_index_data/SiO.txt', 
        n_metal_file='../refractive_index_data/Ag.txt', 
        retardance=True, perfect_mirror=False):

        self.type = 'FlatMirror'
        self.rot_y = rot_y  # rotation in y 
        self.n_film_file = n_film_file
        self.n_film_data = np.genfromtxt(self.n_film_file, delimiter='\t')
        self.n_film_data = self.n_film_data[1:,:]  # remove headers 
        self.film_thickness = film_thickness
        self.n_metal_file = n_metal_file
        self.n_metal_data = np.genfromtxt(self.n_metal_file, delimiter='\t')
        self.n_metal_data = self.n_metal_data[1:,:]
        self.perfect_mirror = perfect_mirror
        self.retardance = retardance  # if false, absolute value of rs and rp used
        # self.delta_x = 0  # rotation in x

    def normalize(self, v):
        norm = np.linalg.norm(v, axis=1).reshape(v.shape[0],1,1)
        norm[norm == 0] = 1
        return v/norm

    def apply_matrix(self, rays, update_history=False):
        ##
        # rotate out of meridional plane
        if rays.isMeridional:  # put back into non meridional basis
            rays.meridional_transform(inverse=True)

        # get N vector
        N = np.array([-np.tan(self.rot_y), 0, -1])
        N = N/np.linalg.norm(N)
        N = N.reshape(1,3,1)

        k_vec = np.real(rays.k_vec)

        print("-----before-----")
        print(rays.k_vec)
        print(rays.theta)
        print(rays.phi)
        # print("k_vec shape", np.shape(rays.k_vec))
        # print("N shape", np.shape(N))

        p = np.cross(rays.k_vec, N, axis=1)  # get p vector (kxN)
        r = np.cross(rays.k_vec, p, axis=1)  # get r vector (kxp)
        
        # basis vectors
        x = np.array([1,0,0]).reshape(1,3,1)
        y = np.array([0,1,0]).reshape(1,3,1)
        z = np.array([0,0,1]).reshape(1,3,1)

        # first rotation matrix:
        m1 = np.cross(z, rays.k_vec, axis=1)
        sin_m1 = np.linalg.norm(m1, axis=1)
        theta_m1 = np.arcsin(sin_m1)
        # print("imaginary kvec comps", np.imag(rays.k_vec))
        # print("sin_m1", sin_m1)
        # print("theta_m1", theta_m1)
        m1_unit = self.normalize(m1)#/np.linalg.norm(m1, axis=1).reshape(m1.shape[0],1,1)
        # print(np.linalg.norm(m1, axis=1).reshape(m1.shape[0],1,1))

        m1_x = m1_unit[:,0]
        m1_y = m1_unit[:,1]
        m1_z = m1_unit[:,2]

        # rotation matrix 1
        M1 = optical_matrices.arbitrary_rotation(theta_m1, m1_x,m1_y,m1_z)
        
        # print("M1",M1)

        # apply 1st rotation matrix
        # print("M1.shape", M1.shape)
        M1 = M1.reshape(M1.shape[0],3,3)
        # print("r.shape", r.shape)
        r_prime = M1 @ r

        # 2nd rotation axis (z -> zk?)
        m2 = np.cross(r_prime, x, axis=1)
        sin_m2 = np.linalg.norm(m2, axis=1)
        theta_m2 = np.arcsin(sin_m2)
        m2_unit = self.normalize(m2)#/np.linalg.norm(m2, axis=1).reshape(m2.shape[0],1,1)

        m2_x = m2_unit[:,0]
        m2_y = m2_unit[:,1]
        m2_z = m2_unit[:,2]
        
        # rotation matrix 2
        M2 = optical_matrices.arbitrary_rotation(theta_m2, m2_x,m2_y,m2_z)
        M2 = M2.reshape(M2.shape[0],3,3)

        # print("sin_m2", sin_m2)
        # print("theta_m2", theta_m2)
        # print("M2",M2)
        
        M2M1 = M2 @ M1
        M2M1_inv = np.linalg.inv(M2M1)
        rays.transfer_matrix = M2M1 @ rays.transfer_matrix

        # Express N in ps space
        N_ps = M2M1 @ N

        # k_tf_mat = M2M1
        k_ps = M2M1 @ rays.k_vec

        # calculate rotation matrix to represent mirror 
        # ray cross N for rotation
        # print(M2M1)
        # print(k_ps)
        # print(N_ps)
        kxN = np.cross(k_ps, N_ps, axis=1)
        sin_mr_1theta = np.linalg.norm(kxN, axis=1)
        theta_i = np.arcsin(sin_mr_1theta)
        # print("kxN", kxN)
        # print("sin_mr_1theta", sin_mr_1theta)
        rot_theta = np.pi-2*theta_i  # rotating from a straight line - 180 - 2theta

        kxN_unit = self.normalize(kxN)#/np.linalg.norm(kxN, axis=1).reshape(kxN.shape[0],1,1)
        kxN_x = kxN_unit[:,0]
        kxN_y = kxN_unit[:,1]
        kxN_z = kxN_unit[:,2]

        M_rotation = optical_matrices.arbitrary_rotation(rot_theta, kxN_x, kxN_y, kxN_z)
        M_rotation = M_rotation.reshape(M_rotation.shape[0],3,3)

        # print(rays.lda)

        if self.perfect_mirror:
            M_fresnel = np.identity(3)
        else:
            # print("theta_i", theta_i)
            M_fresnel = optical_matrices.protected_mirror_fresnel_matrix(
                theta_i, self.n_film_data, self.film_thickness, self.n_metal_data, rays.lda)
        rays.transfer_matrix = M_fresnel @ rays.transfer_matrix

        # compute ray rotation
        rays.transfer_matrix = M_rotation @ rays.transfer_matrix
        rays.transfer_matrix = M2M1_inv @ rays.transfer_matrix  # for E
        k_ps = M_rotation @ k_ps
        rays.k_vec = M2M1_inv @ k_ps
        

        # back into cartesian, now recalculate theta and phi
        # floating point values sometimes mean k_vec[2] > 1 e.g. 1 + 1e-10        
        kz_gt_1_mask = np.squeeze(rays.k_vec[:,2] > 1)
        rays.theta[kz_gt_1_mask] = 0
        if rays.k_vec.shape[0] < 2:  # very annoying edge case when size is 1,3,1 still don't really understand it
            #if kz_gt_1_mask:
            kz_gt_1_mask = np.reshape(kz_gt_1_mask, (1,))
        rays.k_vec[kz_gt_1_mask, 2] = 1
        rays.theta = np.arccos(rays.k_vec[:,2]).flatten()
        rays.phi = np.arctan2(rays.k_vec[:,1], rays.k_vec[:,0]).flatten()
        rays.phi = (rays.phi + 2*np.pi) % (2*np.pi)

        print("------after------")
        print(rays.k_vec)
        print(rays.theta)
        print(rays.phi)
        if update_history: rays.update_history()

        return rays

class ProtectedFlatMirror(FlatMirror):
    def __init__(self, rot_y):
        super().__init__(rot_y)