import warnings
from copy import deepcopy
import numpy as np
import scipy.interpolate as interp
from matplotlib import pyplot as plt
import matplotlib
import os
from math import ceil

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
            
            # avoid negative values from floating point error? later me: "this makes no sense"
            lens_theta[np.abs(lens_theta) < 1e-9] = 0  

            # if abs(rays.rho) > self.D/2:  # Ray escapes system
            # escape_mask_na = abs(np.sin(rays.theta)) > abs(self.sine_theta)
            # now do things with angle -- avoids wrap-around issue hopefully?
            escape_mask_na = abs(rays.theta) > np.arcsin(self.sine_theta)  # replace the 
            rays.escaped = np.logical_or(escape_mask_na, rays.escaped)
            if any(escape_mask_na):
                print(np.sum(escape_mask_na), "escaped from NA mask")
                # print("sine_theta", self.sine_theta)
                # print("max sin(theta)", np.max(np.sin(np.abs(rays.theta))))

            # plt.figure()
            # plt.scatter(rays.phi, rays.theta)
            # plt.show()
            # plt.figure()
            # plt.scatter(rays.phi[np.invert(rays.escaped)], rays.theta[np.invert(rays.escaped)])
            # plt.show()

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
    
class DiagonalMatrix():
    def __init__(self, value, update_history=False):
        self.type = 'DiagonalMatrix'
        self.value = value  # 
        self.update_history = update_history

    def apply_matrix(self, rays, update_history=False):
        if self.update_history: rays.update_history()
        if rays.isMeridional:  # put back into non meridional basis
            rays.meridional_transform(inverse=True)
        rays.transfer_matrix = \
            optical_matrices.diagonal(self.value) @ rays.transfer_matrix
        return rays

class WavePlate():
    def __init__(self, psi, delta, update_history=False,plot_debug=False):
        self.type = 'WavePlate'
        self.psi = psi  # angle of fast axis from x axis
        self.delta = delta  # amount of retardation e.g delta=pi/2 for qwp
        self.update_history = update_history
        self.plot_debug = plot_debug

    def apply_matrix(self, rays, update_history=False):
        if self.update_history: rays.update_history()
        if rays.isMeridional:  # put back into non meridional basis
            rays.meridional_transform(inverse=True)
        # rays.k_vec = optical_matrices.wave_plate(self.psi, self.delta) @ rays.k_vec
        # E_before = rays.transfer_matrix @ rays.E_vec
        #rays_, E_vec_ = 
        if self.plot_debug:
            print("----------------------Electric field before waveplate------------------")
            rays.quiver_plot(downsampling=1, n_rays = None)
        rays.transfer_matrix = \
            optical_matrices.wave_plate(self.psi, self.delta) @ rays.transfer_matrix
        if self.plot_debug:
            print("----------------------Electric field after waveplate------------------")
            rays.quiver_plot(downsampling=1, n_rays = None)    

            x0_all = rays.rho*np.cos(rays.phi)
            y0_all = rays.rho*np.sin(rays.phi)
            x0 = x0_all[np.invert(rays.escaped)]
            y0 =y0_all[np.invert(rays.escaped)]
            
            E_vec = rays.transfer_matrix @ rays.E_vec
            data_x =  np.real(E_vec[0,:,0]*np.conj(E_vec[0,:,0]))
            data_y =  np.real(E_vec[0,:,1]*np.conj(E_vec[0,:,1]))
            graphics.heatmap_plot(x0, y0, data_x, data_y, title="Intensity field after QWP")

        return rays

class IdealFlatMirrorNoRotation():
    pass

class FlatMirror():
    """
    Flat mirror with rotation about y axis
    TODO: make fast version of this without all the tracing and such..
    """
    def __init__(self, rot_y, film_thickness=100e-9, 
        n_film_file='../refractive_index_data/SiO2.txt', 
        n_metal_file='../refractive_index_data/Ag.txt', 
        retardance=True, perfect_mirror=False, update_history=False,
        fresnel_debug_savedir = None,
        reflectance=1,
        plot_debug=False):

        self.type = 'FlatMirror'
        self.mirror_type = 'perfect'  # e.g. fresnel, protected
        self.rot_y = rot_y  # rotation in y 
        self.n_film_file = n_film_file
        self.n_film_data = np.genfromtxt(self.n_film_file, delimiter='\t')
        self.n_film_data = self.n_film_data[1:,:]  # remove headers 
        self.film_thickness = film_thickness
        self.n_metal_file = n_metal_file
        self.n_metal_data = np.genfromtxt(self.n_metal_file, delimiter='\t')
        self.n_metal_data = self.n_metal_data[1:,:]
        self.perfect_mirror = perfect_mirror
        self.reflectance = reflectance
        self.retardance = retardance  # if false, absolute value of rs and rp used
        # self.delta_x = 0  # rotation in x
        self.fresnel_debug_savedir = fresnel_debug_savedir
        self.update_history = update_history
        self.plot_debug = plot_debug

    def normalize(self, v, axis=1):
        norm = np.linalg.norm(v, axis=axis).reshape(v.shape[0],1,1)
        norm[norm == 0] = 1
        return v/norm
    
    def heatmap_plot(self, x0, y0, data_x, data_y, title=""):
        fig = plt.figure(figsize=[10,3]) 

        if len(x0) < 4:
            print("Insufficient points to plot heatmap")
            return

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

        fig.suptitle(title)

    def apply_matrix(self, rays):
        if self.plot_debug:
            print("----------------------Electric field before reflection------------------")
            rays.quiver_plot(downsampling=1, n_rays = None)
        ##
        # rotate out of meridional plane
        if rays.isMeridional:  # put back into non meridional basis
            rays.meridional_transform(inverse=True)
        
        if self.update_history: rays.update_history()

        initial_phi = rays.phi
        if self.plot_debug:
            # starting field
            initial_E_vec_4d = (rays.transfer_matrix @ rays.E_vec)#.squeeze()
            initial_E_vec = initial_E_vec_4d[0,:,:,0]
        
        rho_0 = rays.rho_before_trace
        dist_r = abs(rho_0/np.sin(rays.theta))  # distance for plotting
        # all r are the same, use for theta =0
        dist_r[rays.theta < 1e-6] = dist_r[-1]

        ################################################################
        # plot triangulated heatmap (initial field) without using phi and stuff for debugging
        x0 = rho_0*np.cos(rays.phi)
        y0 = rho_0*np.sin(rays.phi)
        z0 = abs(dist_r)*(1-np.cos(rays.theta))
        if self.plot_debug:

            data_x =  np.real(initial_E_vec[:,0]*np.conj(initial_E_vec[:,0]))
            data_y =  np.real(initial_E_vec[:,1]*np.conj(initial_E_vec[:,1]))
            
            self.heatmap_plot(x0, y0, data_x, data_y, title="Initial field")

        ################################################################

        k_vec_norm = rays.k_vec/np.linalg.norm(rays.k_vec, axis=1).reshape(rays.k_vec.shape[0], 1, 1)
        # print("k_vec_norm", k_vec_norm)
        p0 = np.array([x0, y0, z0]).T
        p_contact = p0 + (dist_r.reshape(dist_r.shape[0],1)*k_vec_norm.squeeze())

        print(self.rot_y)
        # get N vector
        N = np.array([-np.tan(self.rot_y), 0, -1])
        N = N/np.linalg.norm(N)
        N = N.reshape(1,3,1)

        p = np.cross(k_vec_norm, N, axis=1)  # get p vector (kxN) (s wave comp unit?)#
        kdotN = np.sum(k_vec_norm * N, 1)
        # print("kdotN", kdotN)
        np.abs(kdotN.squeeze())  

        r = np.cross(k_vec_norm, p, axis=1)  # get r vector (kxp) (p wave comp unit?)

        # normalize since we compute the angles without the normalization factor...
        p = self.normalize(p)  
        r = self.normalize(r)

        ###################################################################################################
        #### ------------------------------------New method------------------------------------------- ####
        parallel = r[:,:,0]
        senkrecht = p[:,:,0]

        # print("parallel", parallel)

        ps_project = optical_matrices.ps_projection_matrix(parallel, senkrecht, np.squeeze(rays.k_vec))
        ps_project_inv2 = optical_matrices.ps_projection_matrix(parallel, senkrecht, np.squeeze(rays.k_vec), inverse=False)

        M_fresnel_test = np.array([
            [0.95,0,0],
            [0,0.8,0],
            [0,0,0]
        ])
        # print(ps_project)
        inv_ps_proj = np.linalg.inv(ps_project)
        # print("equal invs?", np.array_equal(ps_project_inv2, inv_ps_proj))
         #print(ps_project_inv2-ps_project_inv2)
        #### ----------------------------------------------------------------------------------------- ####
        ###################################################################################################

        # basis vectors
        x = np.array([1,0,0]).reshape(1,3,1)
        y = np.array([0,1,0]).reshape(1,3,1)
        z = np.array([0,0,1]).reshape(1,3,1)
        
        # get angle 
        kxN = np.cross(k_vec_norm, N, axis=1)
        sin_mr_1theta = np.linalg.norm(abs(kxN), axis=1)
        theta_i = np.arcsin(sin_mr_1theta)

        if self.perfect_mirror:
            M_fresnel = np.identity(3)
            M_fresnel *= self.reflectance
            print("USING PERFECT MIRROR")
        else:
            M_fresnel = np.array([
                [1,0,0],
                [0,1,0],
                [0,0,0]
            ])
            print("USING AIRY SUM MIRROR")
            M_fresnel = optical_matrices.protected_mirror_fresnel_matrix(
                theta_i, self.n_film_data, self.film_thickness, self.n_metal_data, rays.lda)
            # print(np.abs(M_fresnel))


        reflection_mat = optical_matrices.reflection_cartesian_matrix(N.squeeze())

        rays.transfer_matrix = reflection_mat @ inv_ps_proj @ M_fresnel @ ps_project @ rays.transfer_matrix 
        k_vec_ref =  reflection_mat @ rays.k_vec
        rays.k_vec = k_vec_ref


        print("----------------------Electric field after reflection------------------")
        if self.plot_debug:
            rays.quiver_plot(downsampling=1, n_rays = None)

        if self.plot_debug:
            # saving E fields for debugging # -------------------------

            E_in = initial_E_vec
            E_ps_proj = ps_project @ initial_E_vec_4d
            E_ps_in = E_ps_proj[0,:,:,0]
            E_ps_out = M_fresnel @ E_ps_proj
            E_ps_out = E_ps_out[0,:,:,0]
            E_out_4d = reflection_mat @ inv_ps_proj @ M_fresnel_test @ E_ps_proj
            E_out = E_out_4d[0,:,:,0]

            if self.fresnel_debug_savedir and not self.perfect_mirror:
                np.savetxt(os.path.join(self.fresnel_debug_savedir, "E_ps_in.csv"), E_ps_in)
                np.savetxt(os.path.join(self.fresnel_debug_savedir, "E_ps_out.csv"), E_ps_out)
                np.savetxt(os.path.join(self.fresnel_debug_savedir, "E_in.csv"), E_in)
                np.savetxt(os.path.join(self.fresnel_debug_savedir, "E_out.csv"), E_out)
                np.savetxt(os.path.join(self.fresnel_debug_savedir, "E_ps_out.csv"), E_ps_out)
                np.savetxt(os.path.join(self.fresnel_debug_savedir, "theta.csv"), theta_i)
                np.savetxt(os.path.join(self.fresnel_debug_savedir, "M_fresnel.csv"), M_fresnel.reshape(M_fresnel.shape[0], 9))

            ##### Check dot product
            Eout_dot = np.sum(k_vec_ref[:,:,0] * E_out, axis=1)
            # print("Dot product out")
            plt.figure()
            plt.scatter(initial_phi, Eout_dot)
            plt.title("Phi against Eout dot with kout")
            plt.show()

        ############   ############################################
        ############ CHECK THIS ############################################
        # rays.k_vec[:,2] = -rays.k_vec[:,2]  # dont worry about handedness? necessary for simulation to work -- otherwise E and k matrices need to be diff?

        # now UN-fold the system to play nice with the tranformation matrices #
        #############################################################################################
        # This needs checking, I assume it is okay but need to verify it gets the same result i.e.  #
        # find another way of doing this without unfolding and compare the result... or manually    #
        # trace ray?                                                                                #
        #############################################################################################

        flip_mat = optical_matrices.flip_axis(2)
        rays.transfer_matrix = flip_mat @ rays.transfer_matrix
        rays.k_vec = flip_mat @ rays.k_vec
        rays.negative_kz = True

        #############################################################################################

        kz_gt_1_mask = np.squeeze(rays.k_vec[:,2] > 1)
        rays.theta[kz_gt_1_mask] = 0
        if rays.k_vec.shape[0] < 2:  # very annoying edge case when size is 1,3,1 still don't really understand it
            #if kz_gt_1_mask:
            kz_gt_1_mask = np.reshape(kz_gt_1_mask, (1,))
        rays.k_vec[kz_gt_1_mask, 2] = 1

        rays.theta = np.arccos(rays.k_vec[:,2]).flatten()

        # MINUS ON PHI TO REPRESENT HANDEDNESS CHANGE
        rays.phi = np.arctan2(rays.k_vec[:,1], rays.k_vec[:,0]).flatten()

        # compare polar to cartesian
        x = np.sin(rays.theta)*np.cos(rays.phi)
        y = np.sin(rays.theta)*np.sin(rays.phi)
        z = np.cos(rays.theta)

        ax0 = plt.figure().add_subplot(projection='3d')
        ax0.plot(x,y,z)

        ax1 = plt.figure().add_subplot(projection='3d')
        ax1.plot(rays.k_vec[:,0], rays.k_vec[:,1], rays.k_vec[:,2])

        if self.plot_debug:
            ###############################################################
            # plot triangulated heatmap (reflected field) for debugging
            data_x =  np.real(E_out[:,0]*np.conj(E_out[:,0]))
            data_y =  np.real(E_out[:,1]*np.conj(E_out[:,0]))
            
            self.heatmap_plot(x, y, data_x, data_y, title="Reflected field")

            ################################################################

            k_vec_norm_out = rays.k_vec/np.linalg.norm(rays.k_vec, axis=1).reshape(rays.k_vec.shape[0], 1,1)
            p_out = p_contact + (dist_r.reshape(dist_r.shape[0],1)*k_vec_norm_out.squeeze())

            if rays.k_vec.shape[0] > 1:
                p_out = p_out.squeeze()
                p_contact = p_contact.squeeze()

            xline1 = np.array([x0, p_contact[:,0]]).T
            yline1 = np.array([y0, p_contact[:,1]]).T
            zline1 = np.array([z0, p_contact[:,2]]).T

            xline2 = np.array([p_contact[:,0], p_out[:,0]]).T
            yline2 = np.array([p_contact[:,1], p_out[:,1]]).T
            zline2 = np.array([p_contact[:,2], p_out[:,2]]).T


            # Used to check inverse transform from p,s ########

            Ep = np.tile([1,0,0], (k_vec_norm.shape[0], 1))
            Es = np.tile([0,1,0], (k_vec_norm.shape[0], 1))
            kps = np.tile([0,0,1], (k_vec_norm.shape[0], 1))
            Ep_ps = np.expand_dims(Ep, axis=[0,3])
            Es_ps = np.expand_dims(Es, axis=[0,3])
            kps = np.expand_dims(kps, axis=[0,3])

            Ep_xyz = inv_ps_proj @ Ep_ps
            Es_xyz = inv_ps_proj @ Es_ps
            Ep_xyz = Ep_xyz[0,:,:,0]
            Es_xyz = Es_xyz[0,:,:,0]

            ###################################################


            ax = plt.figure().add_subplot(projection='3d')

            # for n in range(xline1.shape[0]):
            from matplotlib.pyplot import cm
            color = cm.rainbow(np.linspace(0, 1, 10))
            n_rays = 10
            len_line = len(xline1)
            idxs = range(0,len_line,ceil(len_line/10))
            if len(idxs)>n_rays:
                idxs = idxs[0:-1]
            for i, n in enumerate(idxs):
                ax.plot(xline1[n], yline1[n], zline1[n], linestyle='solid', color=color[i])#, color='green')
                ax.plot(xline2[n], yline2[n], zline2[n], linestyle='dashed', color=color[i])#, color='blue')

                # coord axis
                ax.quiver(x0[n], y0[n], z0[n], Ep_xyz[n,0], Ep_xyz[n,1], Ep_xyz[n,2], length=0.001, color='blue')
                ax.quiver(x0[n], y0[n], z0[n], Es_xyz[n,0], Es_xyz[n,1], Es_xyz[n,2], length=0.001, color='blue')

                ax.quiver(p_contact[n,0],p_contact[n,1], p_contact[n,2], p[n,0], p[n,1], p[n,2],color=color[i], length=0.001)
                ax.quiver(p_contact[n,0],p_contact[n,1], p_contact[n,2], r[n,0], r[n,1], r[n,2],color=color[i], length=0.001)
                ax.set_aspect('equal')

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

