import warnings
from copy import deepcopy
import numpy as np
import scipy.interpolate as interp
from matplotlib import pyplot as plt
import matplotlib
import os

from .tools import graphics
from . import optical_matrices

# Base class for all elements
# I include the possibiliy of off-axis elements and rotated about x even though
# this is unlikely to be supported (maybe rotations will)

class Element():
    def __init__(self):
        self.type = 'Empty element'
        self.update_history = False

    def apply_matrix(self, rays):
        rays.update_history()
        return np.identity(3)

class SineLens(Element):
    """Ideal lens that meets Abbe sine condition"""
    def __init__(self, NA, focal_length, n=1, yAxis_rotation = 0, binning_method=False,
            D=None, show_plots=False, trace_after=True, update_history=False):
        self.type = 'SineLens'
        self.focal_length = focal_length
        self.NA = NA  # we use effective NA (as if lens were in air)
        self.sine_theta = NA/n
        self.n = n
        self.yAxis_rotation = yAxis_rotation
        # used to determine if we stop ray tracing immediately after first surface
        self.binning_method = binning_method
        self.show_plots = show_plots
        self.trace_after = trace_after
        self.update_history = update_history

        if D is None:
            self.D = 2*focal_length*NA*n
        elif abs(D - 2*focal_length*NA*n) > 1e-6*D:
            raise ValueError("D, NA and f are not in agreement")
        else:
            self.D = D

    def apply_matrix(self, rays):
        if self.update_history: rays.update_history()

        if any(np.isnan(rays.phi[np.invert(rays.escaped)])):
            raise Exception("NaN(s) in phi after ray rotation in objective")
        if self.yAxis_rotation != 0:
            print("rotating rays by", self.yAxis_rotation)
            # print("before rotating theta", rays.theta)
            # print("before rotating rho" , rays.rho)
            self.rotate_ray_y(rays)  # for angled lenses i.e. O3
            if self.update_history: rays.update_history(("x_axis rotation %.2f rads before lens" % self.yAxis_rotation))
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
                # print("rho:", rays.rho)
                # print("f:", self.focal_length)
                # print("rho/f:", rays.rho/self.focal_length)

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
            
            rays.rho_before_trace = rays.rho
            if self.trace_after:
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
            # print("rho before", rays.rho)
            self.trace_f(rays) # trace to first surface
            # print(("rho after", rays.rho))
            # print("f", self.focal_length)
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

            # lens_theta = np.arcsin(rays.rho/self.focal_length)  # positive is anticlockwise rotation
            lens_theta = rays.theta # ... or just set angles to zero explicitly
            if any(np.isnan(lens_theta)):
                warnings.warn("NaN values in lens theta - check that rho is not greater than f")
                # print("rho:", rays.rho)
                # print("f:", self.focal_length)
                # print("rho/f:", rays.rho/self.focal_length)

            # plt.figure()
            # plt.plot(rays.theta)
            # plt.title("theta before refraction")
            # plt.show()
            # plt.figure()
            # plt.plot(np.sin(rays.theta))
            # plt.title("sine theta before refraction")
            # plt.show()
            # plt.figure()
            # plt.plot(rays.rho)
            # plt.title("actual rho")
            # plt.show()

            new_theta = rays.theta - lens_theta
            # plt.figure()
            # plt.plot(new_theta)
            # plt.title(" theta after refraction")
            # plt.show()
            
            lens_theta[np.abs(lens_theta) < 1e-9] = 0  # avoid negative values from floating point error

            # if abs(rays.rho) > self.D/2:  # Ray escapes system
            escape_mask_na = abs(np.sin(rays.theta)) > abs(self.sine_theta)
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
    def __init__(self, psi, dz=0, update_history=False):
        self.type = 'LinearPolariser'
        self.psi = psi
        self.dz = dz
        self.update_history = update_history

    def apply_matrix(self, rays, update_history=False):
        # supply update
        if self.update_history: rays.update_history()
        if rays.isMeridional:  # put back into non meridional basis
            rays.meridional_transform(inverse=True)
        # rays.k_vec = optical_matrices.polariser(self.psi) @ rays.k_vec
        rays.transfer_matrix = optical_matrices.polariser(self.psi) @ rays.transfer_matrix
        return rays

class WavePlate():
    def __init__(self, psi, delta, update_history=False):
        self.type = 'WavePlate'
        self.psi = psi  # angle of fast axis from x axis
        self.delta = delta  # amount of retardation e.g delta=pi/2 for qwp
        self.update_history = update_history

    def apply_matrix(self, rays, update_history=False):
        if self.update_history: rays.update_history()
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
    def __init__(self, rot_y, film_thickness=100e-9, 
        n_film_file='../refractive_index_data/SiO.txt', 
        n_metal_file='../refractive_index_data/Ag.txt', 
        retardance=True, perfect_mirror=False, update_history=False,
        fresnel_debug_savedir = None):

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
        self.fresnel_debug_savedir = fresnel_debug_savedir
        self.update_history = update_history

    def normalize(self, v):
        norm = np.linalg.norm(v, axis=1).reshape(v.shape[0],1,1)
        norm[norm == 0] = 1
        return v/norm

    def apply_matrix(self, rays):
        ##
        # rotate out of meridional plane
        if rays.isMeridional:  # put back into non meridional basis
            rays.meridional_transform(inverse=True)
        
        if self.update_history: rays.update_history()

        # starting field
        initial_E_vec_4d = (rays.transfer_matrix @ rays.E_vec)#.squeeze()
        # print(initial_E_vec.shape)
        initial_E_vec = initial_E_vec_4d[0,:,:,0]
        # print(initial_E_vec.shape)

        # initial_E_vec = np.sum(initial_E_vec, axis=0)
        
        # dist_r = 1/np.cos(rays.theta)
        # dist_r = np.ones_like(rays.theta)
        # rho_0 = np.tan(rays.theta)
        rho_0 = rays.rho_before_trace
        z_dist = abs(rho_0/np.tan(rays.theta))
        dist_r = abs(z_dist/np.cos(rays.theta))
        dist_r = abs(rho_0/np.sin(rays.theta))
        # all r are the same, use for theta =0
        z_dist[rays.theta < 1e-6] = dist_r[-1]
        dist_r[rays.theta < 1e-6] = dist_r[-1]

        # print("z_dist", z_dist)
        # print("dist_r", dist_r)
        # print("shape dist", dist_r.shape)
        
        x0 = rho_0*np.cos(rays.phi)
        y0 = rho_0*np.sin(rays.phi)
        z0 = abs(dist_r)*(1-np.cos(rays.theta))
        # z0 = np.zeros_like(rays.phi)

        ################################################################
        # plot triangulated heatmap (initial field) without using phi and stuff for debugging
        
        fig = plt.figure(figsize=[10,3]) 

        data_x =  initial_E_vec[:,0]**2
        data_y =  initial_E_vec[:,1]**2

        max_for_scale = np.max([np.max(data_x), np.max(data_y)])
        min_for_scale = np.min([np.min(data_x), np.min(data_y)])

        import matplotlib
        cbar_ticks = np.linspace(min_for_scale, max_for_scale, 11)
        cmap=matplotlib.cm.get_cmap('autumn_r')

        ax = fig.add_subplot(131)
        # print("I-field data",  trace_E_vec[:,0]**2)
        levels = np.linspace(min_for_scale, max_for_scale, 257)


        pc1 = ax.tricontourf(x0, y0, data_x, cmap=cmap,\
            levels=levels, vmin=min_for_scale,vmax=max_for_scale, extend='both')

        # plt.plot(r_line*np.cos(phi_line), r_line*np.sin(phi_line), color='k', zorder=2)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title("X component intensity")
        
        fig.colorbar(pc1, ax=ax, fraction=0.04, pad=0.15, ticks=cbar_ticks)
        
        ax2 = fig.add_subplot(132)
        levels = np.linspace(min_for_scale, max_for_scale, 257)

        pc2 = ax2.tricontourf(x0, y0, data_y, cmap=cmap,\
            levels=levels, vmin=min_for_scale,vmax=max_for_scale, extend='both')

        # plt.plot(r_line*np.cos(phi_line), r_line*np.sin(phi_line), color='k', zorder=2)
        ax2.set_aspect('equal')
        ax2.axis('off')
        ax2.set_title("Y component intensity")
        
        fig.colorbar(pc2, ax=ax2, fraction=0.04, pad=0.15, ticks=cbar_ticks)
        ax3 = fig.add_subplot(133)
        levels = np.linspace(min_for_scale, max_for_scale, 257)

        pc3 = ax3.tricontourf(x0, y0, data_y+data_x, cmap=cmap,\
            levels=levels, vmin=min_for_scale,vmax=max_for_scale, extend='both')

        # plt.plot(r_line*np.cos(phi_line), r_line*np.sin(phi_line), color='k', zorder=2)
        ax3.set_aspect('equal')
        ax3.axis('off')
        ax3.set_title("Y+X component intensity")
        
        fig.colorbar(pc3, ax=ax3, fraction=0.04, pad=0.15, ticks=cbar_ticks)

        fig.suptitle("E field in")

        ################################################################

        # print("rays.k_vec norm", np.linalg.norm(rays.k_vec, axis=1))
        k_vec_norm = rays.k_vec/np.linalg.norm(rays.k_vec, axis=1).reshape(rays.k_vec.shape[0], 1, 1)

        # test arrays to see if the order is preserved
        k_track_x = np.linspace(0,1,rays.k_vec.shape[0])
        k_track_y = np.linspace(1,0,rays.k_vec.shape[0])
        k_track_z = np.zeros(rays.k_vec.shape[0])
        k_track = np.array([k_track_x, k_track_y, k_track_z]).T
        k_track = np.expand_dims(k_track,2)
        # print("k_track shape", k_track.shape)
        # print(np.linalg.norm(k_track, axis=1).reshape(rays.k_vec.shape[0], 1, 1).shape)
        k_track = k_track/np.linalg.norm(k_track, axis=1).reshape(rays.k_vec.shape[0], 1, 1)
        E_track = np.zeros((1,rays.k_vec.shape[0],3,1))
        E_track[0,:,:,0] = np.array([k_track_x, k_track_y, k_track_z]).T
        # E_track = np.array([k_track_x, k_track_y, k_track_z]).T

        p0 = np.array([x0, y0, z0]).T
        p_contact = p0 + (dist_r.reshape(dist_r.shape[0],1)*k_vec_norm.squeeze())
        # p_contact = p0 + (0.005*rays.k_vec.T/np.linalg.norm(rays.k_vec)).squeeze()

        x0t = np.zeros_like(rays.phi)
        y0t = np.zeros_like(rays.phi)
        z0t = np.zeros_like(rays.phi)
        p0_2 = np.array([x0t, y0t, z0t]).reshape(x0.shape[0], 3)
        p0_2 = p0
        #ax0 = plt.figure().add_subplot(projection='3d')
        #ax1 = plt.figure().add_subplot(projection='3d')
        p_contact_2 = p0_2 + dist_r.reshape(dist_r.shape[0], 1)*(rays.k_vec.squeeze())

        # print("rays.k_vec", rays.k_vec)

        # print("p_contact_2", p_contact_2)

        # print("p0", p0)
        # print("p0_2", p0_2)

        # for i in range(p_contact_2.shape[0]):
        #     ax0.plot([p0_2[i,0], p_contact_2[i,0]], [p0_2[i,1], p_contact_2[i,1]], [p0_2[i,2], p_contact_2[i,2]])
        #     ax0.scatter(p0_2[i,0], p0_2[i,1], p0_2[i,2])
            # print([p0_2[i,0], p_contact_2[i,0]], [p0_2[i,1], p_contact_2[i,1]], [p0_2[i,2], p_contact_2[i,2]])

        # for i in range(p_contact_2.shape[0]):
        #     ax1.scatter(rays.k_vec.squeeze()[i,0], rays.k_vec.squeeze()[i,1], rays.k_vec.squeeze()[i,2])
            # print([rays.k_vec.squeeze()[i,0], rays.k_vec.squeeze()[i,1], rays.k_vec.squeeze()[i,2]])
        #p_contact = p0 + (dist_r*rays.k_vec.T).squeeze()


        print(self.rot_y)
        # get N vector
        N = np.array([-np.tan(self.rot_y), 0, -1])
        N = N/np.linalg.norm(N)
        N = N.reshape(1,3,1)
        # print("N", N)
        k_vec = np.real(rays.k_vec)

        # print("-----before-----")
        # print(rays.k_vec)
        # print("rays vector mag", np.linalg.norm(rays.k_vec, axis=1))
        # print(rays.theta)
        # print(rays.phi)
        # print("k_vec shape", np.shape(rays.k_vec))
        # print("N shape", np.shape(N))

        p = np.cross(k_vec_norm, N, axis=1)  # get p vector (kxN)
        r = np.cross(k_vec_norm, p, axis=1)  # get r vector (kxp)

        # normalize since we compute the angles without the normalization factor...
        p = self.normalize(p)  
        r = self.normalize(r)

        print("R shape", r.shape)
        
        # basis vectors
        x = np.array([1,0,0]).reshape(1,3,1)
        y = np.array([0,1,0]).reshape(1,3,1)
        z = np.array([0,0,1]).reshape(1,3,1)

        # first rotation matrix:
        m1 = np.cross(z, k_vec_norm, axis=1)
        # m1 = np.cross(k_vec_norm, z, axis=1)
        sin_m1 = np.linalg.norm(m1, axis=1)
        theta_m1 = np.arcsin(sin_m1)
        # print("imaginary kvec comps", np.imag(rays.k_vec))
        # print("sin_m1", sin_m1)
        # print("theta_m1", theta_m1)
        m1_unit = self.normalize(m1)#/np.linalg.norm(m1, axis=1).reshape(m1.shape[0],1,1)
        # print(np.linalg.norm(m1, axis=1).reshape(m1.shape[0],1,1))
        print("m1", m1)

        m1_x = m1_unit[:,0]
        m1_y = m1_unit[:,1]
        m1_z = m1_unit[:,2]

        # rotation matrix 1
        M1 = optical_matrices.arbitrary_rotation(theta_m1, m1_x,m1_y,m1_z)
        M1_neg = optical_matrices.arbitrary_rotation(-theta_m1, m1_x,m1_y,m1_z)
        
        # apply 1st rotation matrix
        # print("M1.shape", M1.shape)
        M1 = M1.reshape(M1.shape[0],3,3)
        M1_inv = np.linalg.inv(M1)
        M1_neg = M1_neg.reshape(M1_neg.shape[0],3,3)
        # print("r.shape", r.shape)
        r_prime = M1 @ r
        x_prime = M1 @ x
        p_prime = M1 @ p
        z_prime = M1 @ z

        # test to see how Ep Es coord transform looks at different (phi, theta)
        Ep = np.tile([1,0,0], (k_vec_norm.shape[0], 1))
        Es = np.tile([0,1,0], (k_vec_norm.shape[0], 1))
        kps = np.tile([0,0,1], (k_vec_norm.shape[0], 1))
        Ep_ps = np.expand_dims(Ep, axis=[0,3])
        Es_ps = np.expand_dims(Es, axis=[0,3])
        kps = np.expand_dims(kps, axis=[0,3])

        # 2nd rotation axis (z -> zk?)
        rpxXp = np.cross(r_prime, x_prime, axis=1)
        m2 = np.cross(r_prime, x, axis=1)
        # m2 = np.cross(r_prime, x_prime, axis=1)

        print("r' cross x' (should be about k_vec, 0,0,1)", rpxXp)
        print("kvec", k_vec_norm)
        print("zprime", z_prime)
        print("m2", m2)

        sin_m2 = np.linalg.norm(m2, axis=1)
        print("sin_m2", sin_m2)
        theta_m2 = np.arcsin(sin_m2)
        m2_unit = self.normalize(m2)#/np.linalg.norm(m2, axis=1).reshape(m2.shape[0],1,1)

        m2_x = m2_unit[:,0]
        m2_y = m2_unit[:,1]
        m2_z = m2_unit[:,2]
        
        #### ALTERNATIVE M2 USING Ep
        # Ep
        #
        # m2_ep = np.cross(, Ep[:,0], axis=1)

        # rotation matrix 2
        M2 = optical_matrices.arbitrary_rotation(theta_m2, m2_x,m2_y,m2_z)
        M2_neg = optical_matrices.arbitrary_rotation(-theta_m2, m2_x,m2_y,m2_z)

        M2 = M2.reshape(M2.shape[0],3,3)
        M2_neg = M2_neg.reshape(M2_neg.shape[0],3,3)
        M2_inv = np.linalg.inv(M2)

        # print("sin_m2", sin_m2)
        # print("theta_m2", theta_m2)
        # print("M2",M2)
        # TODO: remove this, testing if M2 actually does anything
        # M2 = np.tile(np.identity(3), (M1.shape[0], 1, 1))
        
        M2M1 = M2 @ M1
        M2M1_neg = M2_neg @ M1
        M2M1_inv = np.linalg.inv(M2M1)
        M2M1_neg_inv = np.linalg.inv(M2M1_neg)
        # rays.transfer_matrix = M2M1 @ rays.transfer_matrix

        # Express N in ps space
        N_ps = M2M1 @ N

        # print(k_track.shape)
        # print(rays.k_vec.shape)
        # k_tf_mat = M2M1
        k_ps = M2M1 @ rays.k_vec
        k_ps_m1 = M1 @ rays.k_vec
        k_track_ps = M2M1 @ k_track

        # calculate rotation matrix to represent mirror 
        # ray cross N for rotation
        # print(M2M1)
        # print(k_ps)
        # print("N_ps", N_ps)
        kxN = np.cross(N_ps, k_ps, axis=1)
        # kxN = np.cross(N_ps, k_ps, axis=1)
        sin_mr_1theta = np.linalg.norm(abs(kxN), axis=1)
        theta_i = np.arcsin(sin_mr_1theta)
        # print("theta_i", theta_i)
        # print("kxN", kxN)
        # print("sin_mr_1theta", sin_mr_1theta)
        rot_theta = np.pi-2*theta_i  # rotating from a straight line - 180 - 2theta

        kxN_unit = self.normalize(kxN)#/np.linalg.norm(kxN, axis=1).reshape(kxN.shape[0],1,1)
        kxN_x = kxN_unit[:,0]
        kxN_y = kxN_unit[:,1]
        kxN_z = kxN_unit[:,2]

        M_rotation = optical_matrices.arbitrary_rotation(rot_theta, kxN_x, kxN_y, kxN_z)
        M_rotation = M_rotation.reshape(M_rotation.shape[0],3,3)
        M_rotation_E = optical_matrices.arbitrary_rotation(-2*theta_i, kxN_x, kxN_y, kxN_z)
        M_rotation_E = M_rotation_E.reshape(M_rotation_E.shape[0],3,3)
        M_rotation_coord = optical_matrices.arbitrary_rotation(-rot_theta, kxN_x, kxN_y, kxN_z)
        M_rotation_coord = M_rotation_coord.reshape(M_rotation_coord.shape[0],3,3)

        M2M1_rot_inv = np.linalg.inv(M_rotation_coord @ M2M1)
        # compare to simple method:
        # k_vec_ref = rays.k_vec - 2*(rays.k_vec.reshape(rays.k_vec.shape[0], 3).dot(N.reshape(1,3)))*N#
        # print("dot shape", np.sum(rays.k_vec * N ,axis=1).flatten().shape)
        vec_change = abs(2*np.sum(k_vec_norm * N ,axis=1).flatten()*N).T

        reflection_mat = optical_matrices.reflection_cartesian_matrix(N.squeeze())

        # print(vec_change)
        # print((2*np.sum(rays.k_vec * N ,axis=1).flatten()*N).T)
        # print("vec_change shape", vec_change.shape)
        # print("rays.k_vec shape", rays.k_vec.shape)
        k_vec_ref_2 = k_vec_norm - vec_change
        # print(k_vec_ref.shape)

        if self.perfect_mirror:
            M_fresnel = np.identity(3)
        else:
            # print("theta_i", theta_i)
            M_fresnel = optical_matrices.protected_mirror_fresnel_matrix(
                theta_i, self.n_film_data, self.film_thickness, self.n_metal_data, rays.lda)
        # rays.transfer_matrix = M_fresnel @ rays.transfer_matrix

        # invert z component to reverse direction/unfold system, is this valid?
        # rays.k_vec[:, 2] = -rays.k_vec[:, 2]
        reverse_mat = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])

        print(reflection_mat)
        k_vec_ref =  reverse_mat @ reflection_mat @ rays.k_vec

        E_ps_in = M2M1 @ initial_E_vec_4d
        E_ps_out = M2M1_inv @ M_fresnel @ M2M1 @ initial_E_vec_4d
        E_ps = M2M1 @ initial_E_vec_4d
        E_ps = E_ps[0,:,:,0]
        E_ps_in = E_ps_in[0,:,:,0]
        E_ps_out = E_ps_out[0,:,:,0]

        print(Es.shape)
        Ep = M2M1_inv @ Ep_ps
        Es = M2M1_inv @ Es_ps
        # Ep = M2M1 @ Ep_ps
        # Es = M2M1 @ Es_ps
        kps = M2M1_inv @ kps
        
        Ep_zk = M1_inv @ Ep_ps
        Es_zk = M1_inv @ Es_ps

        # Ep_zk = M1_inv @ M2_inv @ Ep
        # Es_zk = M1_inv @ M2_inv @ Es

        Epinv = M2M1 @ Ep
        Esinv = M2M1 @ Es
        print(Es.shape)
        Ep_xyz = Ep[0,:,:,0]
        Es_xyz = Es[0,:,:,0]
        Ep_xyz_zk = Ep_zk[0,:,:,0]
        Es_xyz_zk = Es_zk[0,:,:,0]
        kps_xyz = kps[0,:,:,0]
        kps = M1 @ rays.k_vec
        print(Es_xyz.shape)

        if self.fresnel_debug_savedir is not None and not self.perfect_mirror:
            np.savetxt(os.path.join(self.fresnel_debug_savedir, "E_ps_in.csv"), E_ps_in)
            np.savetxt(os.path.join(self.fresnel_debug_savedir, "E_ps_out.csv"), E_ps_out)
            np.savetxt(os.path.join(self.fresnel_debug_savedir, "theta.csv"), theta_i)
            np.savetxt(os.path.join(self.fresnel_debug_savedir, "M_fresnel.csv"), M_fresnel.reshape(M_fresnel.shape[0], 9))

            plt.figure()
            plt.plot(theta_i, M_fresnel[:,0,0])
            plt.plot(theta_i, M_fresnel[:,1,1])
            plt.show()

            fig = plt.figure(figsize=[10,3]) 

            data_x =  np.real(E_ps[:,0]*np.conj(E_ps[:,0]))
            data_y =  np.real(E_ps[:,1]*np.conj(E_ps[:,1]))
            data_z =  np.real(E_ps[:,2]*np.conj(E_ps[:,2]))

            # print("Iz for Eps (shold be zero)", data_z)

            max_for_scale = np.max([np.max(data_x), np.max(data_y)])
            min_for_scale = np.min([np.min(data_x), np.min(data_y)])

            
            cbar_ticks = np.linspace(min_for_scale, max_for_scale, 11)
            cmap=matplotlib.cm.get_cmap('autumn_r')

            ax = fig.add_subplot(131)
            # print("I-field data",  trace_E_vec[:,0]**2)
            levels = np.linspace(min_for_scale, max_for_scale, 257)

            # p_out[:,0], p_out[:,1]
            pc1 = ax.tricontourf(x0, y0, data_x, cmap=cmap,\
                levels=levels, vmin=min_for_scale,vmax=max_for_scale, extend='both')

            # plt.plot(r_line*np.cos(phi_line), r_line*np.sin(phi_line), color='k', zorder=2)
            ax.set_aspect('equal')
            ax.axis('off')
            ax.set_title("P component intensity")
            
            fig.colorbar(pc1, ax=ax, fraction=0.04, pad=0.15, ticks=cbar_ticks)
            
            ax2 = fig.add_subplot(132)
            levels = np.linspace(min_for_scale, max_for_scale, 257)

            pc2 = ax2.tricontourf(x0, y0, data_y, cmap=cmap,\
                levels=levels, vmin=min_for_scale,vmax=max_for_scale, extend='both')

            # plt.plot(r_line*np.cos(phi_line), r_line*np.sin(phi_line), color='k', zorder=2)
            ax2.set_aspect('equal')
            ax2.axis('off')
            ax2.set_title("S component intensity")
            
            fig.colorbar(pc2, ax=ax2, fraction=0.04, pad=0.15, ticks=cbar_ticks)
            fig.suptitle("E_p and Es")

        E_vec_ref = reverse_mat @ reflection_mat @ M2M1_inv @ M_fresnel @ M2M1 @ initial_E_vec_4d#rays.E_vec
        rays.transfer_matrix = reverse_mat @ reflection_mat @ M2M1_inv @ M_fresnel @ M2M1 @ rays.transfer_matrix 
        # print("k_ps", k_ps)
        k_before_rot = M2M1_inv @ k_ps

        # print("k_before_rot", k_before_rot)
        fig_kz_beforerot = plt.figure()
        plt.plot(k_before_rot[:,2])
        plt.title("k_before_rot")
        plt.show()

        # print("ref k difference", k_vec_ref - k_vec_ref_2)

        # compute ray rotation
        # rays.transfer_matrix = M_rotation @ rays.transfer_matrix
        # rays.transfer_matrix = M_rotation_E @ rays.transfer_matrix
        # TODO: maybe remove, alternative approach -- coordinate inverse transformation with rotated coord space

        # rays.transfer_matrix = M2M1_inv @ rays.transfer_matrix  # for E
        # rays.transfer_matrix = M2M1_rot_inv @ rays.transfer_matrix

        k_ps = M_rotation @ k_ps
        rays.k_vec = k_vec_ref#M2M1_inv @ k_ps
        # rays.k_vec = M2M1_inv @ k_ps
        k_track_fin = M2M1_inv @ k_track_ps

        scatter_k = plt.figure().add_subplot(projection='3d')
        scatter_k.plot(rays.k_vec[:,0], rays.k_vec[:,1], rays.k_vec[:,2])
        scatter_k.set_title("k reflected")


        # back into cartesian, now recalculate theta and phi
        # floating point values sometimes mean k_vec[2] > 1 e.g. 1 + 1e-10        
        kz_gt_1_mask = np.squeeze(rays.k_vec[:,2] > 1)
        rays.theta[kz_gt_1_mask] = 0
        if rays.k_vec.shape[0] < 2:  # very annoying edge case when size is 1,3,1 still don't really understand it
            #if kz_gt_1_mask:
            kz_gt_1_mask = np.reshape(kz_gt_1_mask, (1,))
        rays.k_vec[kz_gt_1_mask, 2] = 1
        rays.theta = np.arccos(rays.k_vec[:,2]).flatten()
        # rays.theta = np.pi - rays.theta
        # /(rays.k_vec[:,0]**2+rays.k_vec[:,1]**2+rays.k_vec[:,2]**2)**0.5
        # kz plot


        # rays.theta = np.pi-rays.theta
        # rays.theta = rays.theta % np.pi/2
        rays.phi = np.arctan2(rays.k_vec[:,1], rays.k_vec[:,0]).flatten()
        # rays.phi = (rays.phi + 2*np.pi) % (2*np.pi)
        # rays.phi = -rays.phi

        # print("k_vec_ref",k_vec_ref)
        # print("k_vec_ref - rays.k_vec", k_vec_ref - rays.k_vec)

        rays.negative_kz = True

        # draw stuff....
        # draw to surface with random f, e.g. 0.180/60
        # x0 = rays.rho*np.sin(rays.phi)
        # y0 = rays.rho*np.cos(rays.phi)
        # z0 = np.zeros_like(x0)

        # compare polar to cartesian
        x = np.sin(rays.theta)*np.cos(rays.phi)
        y = np.sin(rays.theta)*np.sin(rays.phi)
        z = np.cos(rays.theta)

        ax0 = plt.figure().add_subplot(projection='3d')
        ax0.plot(x,y,z)

        ax1 = plt.figure().add_subplot(projection='3d')
        ax1.plot(rays.k_vec[:,0], rays.k_vec[:,1], rays.k_vec[:,2])

        #print(p0)
        # print(p0.shape)
        # print((dist_r*rays.k_vec.T).squeeze().shape)
        # print(p_contact.shape)

        k_vec_norm_out = rays.k_vec/np.linalg.norm(rays.k_vec, axis=1).reshape(rays.k_vec.shape[0], 1,1)
        p_out = p_contact + (dist_r.reshape(dist_r.shape[0],1)*k_vec_norm_out.squeeze())
        p_out_ref = p_contact + (dist_r.reshape(dist_r.shape[0],1)*k_vec_ref.squeeze())

        p_out = p_out.squeeze()
        p_contact = p_contact.squeeze()
        p_contact = p_contact_2

        # print(x0.shape)
        # print(p_contact.shape)
        # print(p_contact[0].shape)
        x_points = np.array([x0, p_contact[:,0], p_out[:,0]])
        y_points = [y0, p_contact[:, 1], p_out[:, 1]]
        z_points = [z0, p_contact[:,2], p_out[:, 2]]

        x_normal_line = [p_contact[0,0], p_contact[0,0] + dist_r[0]*N[0,0,0]]
        y_normal_line = [p_contact[0,1], p_contact[0,1] + dist_r[0]*N[0,1,0]]
        z_normal_line = [p_contact[0,2], p_contact[0,2] + dist_r[0]*N[0,2,0]]

        print("N", N[0,0,0], N[0,1,0], N[0,2,0])

        # normal_line = np.array([
        #     [p_contact[0,0], p_contact[0,1], p_contact[0,2]],
        #     [p_contact[0,0]+dist_r[0]*N[0,0,0], p_contact[0,1]+dist_r[0]*N[0,1,0], p_contact[0,2]+dist_r[0]*N[0,2,0]]
        # ])

        xline1 = np.array([x0, p_contact[:,0]]).T
        yline1 = np.array([y0, p_contact[:,1]]).T
        zline1 = np.array([z0, p_contact[:,2]]).T

        xline2 = np.array([p_contact[:,0], p_out[:,0]]).T
        yline2 = np.array([p_contact[:,1], p_out[:,1]]).T
        zline2 = np.array([p_contact[:,2], p_out[:,2]]).T

        xline2r = np.array([p_contact[:,0], p_out_ref[:,0]]).T
        yline2r = np.array([p_contact[:,1], p_out_ref[:,1]]).T
        zline2r = np.array([p_contact[:,2], p_out_ref[:,2]]).T

        # trace_E_vec = (rays.transfer_matrix @ rays.E_vec)#.squeeze()
        trace_E_vec = E_vec_ref
        #trace_E_vec = np.sum(trace_E_vec, axis=0)
        #print(trace_E_vec.shape)
        trace_E_vec = trace_E_vec[0,:,:,0]
        trace_E_track = rays.transfer_matrix @ E_track
        trace_E_track = trace_E_track[0,:,:,0]

        print("#################################################")
        # print("k track", k_track_fin)
        # print("trace_E_track", trace_E_track)


        ax = plt.figure().add_subplot(projection='3d')

        #ax.plot(xline1, yline1, zline1)
        #ax.plot(xline2, yline2, zline2)
        # print("shape E trace", trace_E_vec.shape)

        # for n in range(xline1.shape[0]):
        from matplotlib.pyplot import cm
        color = cm.rainbow(np.linspace(0, 1, 10))
        n_rays = 10
        len_line = len(xline1)
        idxs = range(0,len_line,int(len_line/10))
        if len(idxs)>n_rays:
            idxs = idxs[0:-1]
        for i, n in enumerate(idxs):
            ax.plot(xline1[n], yline1[n], zline1[n], linestyle='solid', color=color[i])#, color='green')
            ax.plot(xline2[n], yline2[n], zline2[n], linestyle='dashed', color=color[i])#, color='blue')
        #     ax.plot(xline2r[n], yline2r[n], zline2r[n], color='orange', linestyle='dashed')
            ax.quiver(x0[n], y0[n], z0[n], initial_E_vec[n,0], initial_E_vec[n,1], initial_E_vec[n,2], length=0.001)
            # print("E field", initial_E_vec[0,:])
            ax.quiver(p_out[n,0], p_out[n,1], p_out[n,2], trace_E_vec[n,0], trace_E_vec[n,1], trace_E_vec[n,2], length=0.001)

            # coord axis
            ax.quiver(x0[n], y0[n], z0[n], Ep_xyz[n,0], Ep_xyz[n,1], Ep_xyz[n,2], length=0.001, color='blue')
            ax.quiver(x0[n], y0[n], z0[n], Es_xyz[n,0], Es_xyz[n,1], Es_xyz[n,2], length=0.001, color='blue')
            ax.quiver(x0[n], y0[n], z0[n], Ep_xyz_zk[n,0], Ep_xyz_zk[n,1], Ep_xyz_zk[n,2], length=0.001, color='red')
            ax.quiver(x0[n], y0[n], z0[n], Es_xyz_zk[n,0], Es_xyz_zk[n,1], Es_xyz_zk[n,2], length=0.001, color='red')
            ax.quiver(x0[n], y0[n], z0[n], kps_xyz[n,0], kps_xyz[n,1], kps_xyz[n,2], length=0.001, color='green')
            # ax.quiver(x0[n], y0[n], z0[n], k_ps_m1[n,0], k_ps_m1[n,1], k_ps_m1[n,2], length=0.001, color='green')

            #Ep_xyz_inv

            ax.quiver(p_contact[n,0],p_contact[n,1], p_contact[n,2], p[n,0], p[n,1], p[n,2],color=color[i], length=0.001)
            ax.quiver(p_contact[n,0],p_contact[n,1], p_contact[n,2], r[n,0], r[n,1], r[n,2],color=color[i], length=0.001)
            print("E field", initial_E_vec[n,0], initial_E_vec[n,1], initial_E_vec[n,2])
            ax.set_aspect('equal')
        # ax.scatter(x0, y0, z0)
        Ep_ks_dot = np.sum(kps_xyz * Ep_xyz, axis=1)
        Es_ks_dot = np.sum(kps_xyz * Es_xyz, axis=1)
        print("Ep_ks_dot", Ep_ks_dot)
        print("Es_ks_dot", Es_ks_dot)
        print("Ep_ks_dot just_m1", np.sum(k_vec_norm[:,:,0] * Ep_xyz_zk, axis=1))
        print("Es_ks_dot just_m1", np.sum(k_vec_norm[:,:,0] * Es_xyz_zk, axis=1))
        print("Eps z comp", E_ps_in)
        print("r k dot (should be zero)",  np.sum(k_vec_norm * r, axis=1))


        ax.plot(x_normal_line, y_normal_line, z_normal_line, linewidth=3)
        # scalar triple prod to check if coplanar (a×b)⋅c
        # print(x_normal_line)
        # print(y_normal_line)
        # print(z_normal_line)

        plt.show()

        # plot triangulated heatmap without using phi and stuff for debugging
        
        fig = plt.figure(figsize=[10,3]) 

        data_x =  np.real(trace_E_vec[:,0]*np.conj(trace_E_vec[:,0]))
        data_y =  np.real(trace_E_vec[:,1]*np.conj(trace_E_vec[:,1]))

        max_for_scale = np.max([np.max(data_x), np.max(data_y)])
        min_for_scale = np.min([np.min(data_x), np.min(data_y)])

        
        cbar_ticks = np.linspace(min_for_scale, max_for_scale, 11)
        cmap=matplotlib.cm.get_cmap('autumn_r')

        ax = fig.add_subplot(131)
        # print("I-field data",  trace_E_vec[:,0]**2)
        levels = np.linspace(min_for_scale, max_for_scale, 257)

        # p_out[:,0], p_out[:,1]
        pc1 = ax.tricontourf(x, y, data_x, cmap=cmap,\
            levels=levels, vmin=min_for_scale,vmax=max_for_scale, extend='both')

        # plt.plot(r_line*np.cos(phi_line), r_line*np.sin(phi_line), color='k', zorder=2)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title("X component intensity")
        
        fig.colorbar(pc1, ax=ax, fraction=0.04, pad=0.15, ticks=cbar_ticks)
        
        ax2 = fig.add_subplot(132)
        levels = np.linspace(min_for_scale, max_for_scale, 257)

        pc2 = ax2.tricontourf(x, y, data_y, cmap=cmap,\
            levels=levels, vmin=min_for_scale,vmax=max_for_scale, extend='both')

        # plt.plot(r_line*np.cos(phi_line), r_line*np.sin(phi_line), color='k', zorder=2)
        ax2.set_aspect('equal')
        ax2.axis('off')
        ax2.set_title("Y component intensity")
        
        fig.colorbar(pc2, ax=ax2, fraction=0.04, pad=0.15, ticks=cbar_ticks)
        
        basis_mat = optical_matrices.rotate_basis_tensor(rays.phi, rays.theta)
        print(rays.phi.shape)
        unmix_merid_mat = optical_matrices.meridional_transform_tensor(rays.phi)

        print(basis_mat.shape)
        print(trace_E_vec.shape)
        trace_E_vec_t = np.expand_dims(trace_E_vec, 2)
        trace_E_vec_basis = basis_mat @ trace_E_vec_t
        trace_E_vec_basis = unmix_merid_mat @ trace_E_vec_basis

        trace_E_vec_basis = trace_E_vec_basis[:,:,0]

        ##########################
        figb = plt.figure(figsize=[10,3]) 

        data_x =  np.real(trace_E_vec_basis[:,0]*np.conj(trace_E_vec_basis[:,0]))
        data_y =  np.real(trace_E_vec_basis[:,1]*np.conj(trace_E_vec_basis[:,1]))

        max_for_scale = np.max([np.max(data_x), np.max(data_y)])
        min_for_scale = np.min([np.min(data_x), np.min(data_y)])

        import matplotlib
        cbar_ticks = np.linspace(min_for_scale, max_for_scale, 11)
        cmap=matplotlib.cm.get_cmap('autumn_r')

        ax = figb.add_subplot(131)
        # print("I-field data",  trace_E_vec[:,0]**2)
        levels = np.linspace(min_for_scale, max_for_scale, 257)

        # p_out[:,0], p_out[:,1]
        pc1 = ax.tricontourf(x, y, data_x, cmap=cmap,\
            levels=levels, vmin=min_for_scale,vmax=max_for_scale, extend='both')

        # plt.plot(r_line*np.cos(phi_line), r_line*np.sin(phi_line), color='k', zorder=2)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title("X component intensity")
        
        figb.colorbar(pc1, ax=ax, fraction=0.04, pad=0.15, ticks=cbar_ticks)
        
        ax2 = figb.add_subplot(132)
        levels = np.linspace(min_for_scale, max_for_scale, 257)

        pc2 = ax2.tricontourf(x, y, data_y, cmap=cmap,\
            levels=levels, vmin=min_for_scale,vmax=max_for_scale, extend='both')

        # plt.plot(r_line*np.cos(phi_line), r_line*np.sin(phi_line), color='k', zorder=2)
        ax2.set_aspect('equal')
        ax2.axis('off')
        ax2.set_title("Y component intensity")
        
        figb.colorbar(pc2, ax=ax2, fraction=0.04, pad=0.15, ticks=cbar_ticks)
        figb.suptitle("reflected with basis tf")
        # print("------after------")
        # print(rays.k_vec)
        # print("rays.theta", rays.theta)
        # print(rays.phi)

        """
        rays.finalize_rays_coordinates()
        trace_E_vec2 = (rays.transfer_matrix @ rays.E_vec)
        trace_E_vec2 = trace_E_vec2[0,:,:,0]

        #####################################################
        fig2 = plt.figure(figsize=[10,3]) 

        data_x =  trace_E_vec2[:,0]**2
        data_y =  trace_E_vec2[:,1]**2

        max_for_scale = np.max([np.max(data_x), np.max(data_y)])
        min_for_scale = np.min([np.min(data_x), np.min(data_y)])

        import matplotlib
        cbar_ticks = np.linspace(min_for_scale, max_for_scale, 11)
        cmap=matplotlib.cm.get_cmap('autumn_r')

        ax = fig2.add_subplot(131)
        # print("I-field data",  trace_E_vec[:,0]**2)
        levels = np.linspace(min_for_scale, max_for_scale, 257)


        pc1 = ax.tricontourf(p_out[:,0], p_out[:,1], data_x, cmap=cmap,\
            levels=levels, vmin=min_for_scale,vmax=max_for_scale, extend='both')

        # plt.plot(r_line*np.cos(phi_line), r_line*np.sin(phi_line), color='k', zorder=2)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title("X component intensity finalized")
        
        fig.colorbar(pc1, ax=ax, fraction=0.04, pad=0.15, ticks=cbar_ticks)
        
        ax2 = fig2.add_subplot(132)
        levels = np.linspace(min_for_scale, max_for_scale, 257)

        pc2 = ax2.tricontourf(p_out[:,0], p_out[:,1], data_y, cmap=cmap,\
            levels=levels, vmin=min_for_scale,vmax=max_for_scale, extend='both')

        # plt.plot(r_line*np.cos(phi_line), r_line*np.sin(phi_line), color='k', zorder=2)
        ax2.set_aspect('equal')
        ax2.axis('off')
        ax2.set_title("Y component intensity finalized")
        
        fig.colorbar(pc2, ax=ax2, fraction=0.04, pad=0.15, ticks=cbar_ticks)
        #################################################################
        """
        ## try unfolding...? (Ez = -Ez and kz=-kz)


        # if not rays.isMeridional:
            # print("dot products of ray in merid", np.sum(rays.I_vec * rays.k_vec, axis=1))
            # print("merid after mirror")
            # rays.meridional_transform()
        # plt.hist(rays.theta)

        figkz = plt.figure()
        plt.plot(rays.k_vec[:,2])
        plt.title("kz")
        plt.show()

        # theta plot
        figt = plt.figure()
        plt.plot(rays.theta)
        plt.show()

        return rays

class ProtectedFlatMirror(FlatMirror):
    def __init__(self, rot_y):
        super().__init__(rot_y)

class PerfectRightAngleMirror(Element):
    def __init__(self, update_history):
        super().__init__()
        self.update_history = update_history

    def apply_matrix(self, rays):
        rays.optical_axis += np.pi/2