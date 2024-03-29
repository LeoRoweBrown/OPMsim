from copy import deepcopy
# from tkinter import E
import numpy as np
import copy
from . import optical_matrices
import matplotlib.pyplot as plt
from math import ceil

## vectorised version of ray

class PolarRays:
    def __init__(self,
                phi_array, theta_array, radius,
                area_array, lda=500e-9, keep_history=True, num_rays_saved=None):
        """
        Arguments:
        lda <float> -- wavelength of ray

        Methods:
        """
        # print(phi_array)
        self.n = len(phi_array)
        self.n_final = self.n
        self.lda = lda  # wavelength
        # these are all updated/affected by transforms
        # pad to work with parallel matrix mul in the form of (3,3,N) tensors
        k_vec = np.column_stack(
            (np.sin(theta_array)*np.cos(phi_array),
            np.sin(theta_array)*np.sin(phi_array),
            np.cos(theta_array)))
        
        self.k_vec = np.reshape(k_vec, (self.n, 3, 1))  # n by 3 matrix (x, y, z)
        self.phi = phi_array
        self.theta_in = theta_array  # NOT UPDATED
        self.theta = theta_array
        self.rho = np.zeros_like(phi_array)
        self.rho_before_trace = None
        self.radius = radius
        self.pos = np.zeros((len(phi_array), 3, 1))
        self.optical_axis = 0
        self.isMeridional = False
        self.escaped = [False]*self.n
        self.areas = area_array
        self.area_scaling = np.ones(self.n)
        self.transfer_matrix = np.tile(np.identity(3), (1,self.n,1,1))
        # self.transfer_matrix = self.transfer_matrix.reshape(1,self.n,3,3)
        self.alternative_minimum = None  # used when E-values are set to 0 manually
        self.negative_kz = False
        self.ray_density = self.n/np.sum(area_array)  # so values dont change with ray number
        self.emission_efficiency = 1
        self.half_sphere_energy = 1
        self.average_energy_times_NA = 1

        self.keep_history = keep_history  # actually overrided by apply_matrix which decides this..
        self.num_rays_saved = num_rays_saved

        self.ray_history = []

        if self.num_rays_saved is None:
            self.ray_history_mask = np.arange(0,self.k_vec.shape[0])
        else:
            self.ray_history_mask = np.round(np.linspace(0, self.k_vec.shape[0]-1, self.num_rays_saved))
            self.ray_history_mask = np.array(set(self.ray_history_mask))  # remove duplicates

    def update_history(self, note=None):
        self.note = note
        if self.keep_history:
            copy_to_save = copy.deepcopy(self)
            # TODO: please rename remove_escaped_rays now that it's being used polymorphically
            copy_to_save.remove_escaped_rays(self.ray_history_mask)
            copy_to_save.ray_history = []  # don't want multiple copies of ray history
            self.ray_history.append(copy_to_save)
            print("Saved checkpoint with", self.num_rays_saved, "rays")
        else:
            # no reason to really have this option
            print("History has been disabled, rays not saved!")

    def rotate_rays_local_basis(self, inverse_basis=False, inverse_meridional=True):
        """
        Rotate each E_vec to local frame (defined for ray i by phi[i] and theta[i])
        """
        # check orthogonality of E and k
        # print("dot product", np.sum(self.E_vec * self.k_vec, axis=1))
        # print("basis rotate transfer matrix before", self.transfer_matrix)
        # we want to start from non meridional basis, unless we are doing
        # inverse basis transform (inverse basis tf undoes meridional)
        reverse_phi = False
        theta = self.theta
        if self.isMeridional and not inverse_basis:
            print("Inverting meridional")
            self.meridional_transform(inverse=True)
        # method 1:
        if np.any(self.k_vec[:,2] < 0):
            print("kz are negative!")
            reverse_phi = True
            #theta = np.pi-self.theta
        else:
            pass
            #theta = self.theta
        # method 2:
        # theta = self.theta % (np.pi/2)
        # plt.figure()
        # plt.plot(theta)
        # plt.title("theta before local basis")
        # plt.show()

        basis_tensor = optical_matrices.rotate_basis_tensor(
            self.phi, theta, inverse_basis)
        # thinking about removing k_vec tf, but it's undone in merid so has to stay..
        self.k_vec = basis_tensor @ self.k_vec  
        self.transfer_matrix = basis_tensor @ self.transfer_matrix
        # print("basis rotate transfer matrix", self.transfer_matrix)

        self.isMeridional = True # basis rotation automatically does meridional tf
        if inverse_meridional:
            self.meridional_transform(inverse=True, reverse_phi=reverse_phi)  # recover global Ex and Ey


    def meridional_transform(self, inverse=False, reverse_phi=False):
        if self.isMeridional != inverse:
            raise Exception("Cannot apply meridional transform twice in same direction")
        phi = self.phi
        merid_tensor_rev = optical_matrices.meridional_transform_tensor(
            -self.phi, inverse)     
        merid_tensor = optical_matrices.meridional_transform_tensor(
            self.phi, inverse)
        
        if reverse_phi: # or rays.negative_kz
            self.transfer_matrix = merid_tensor_rev @ self.transfer_matrix
        else:
            self.transfer_matrix = merid_tensor @ self.transfer_matrix
        self.k_vec = merid_tensor @ self.k_vec
        self.isMeridional = not inverse


    def finalize_rays_coordinates(self,inverse_meridional=True):

        if self.isMeridional:  # put back into non meridional basis
            self.meridional_transform(inverse=True)

        self.update_history()  # save rays before
        # print("before rotate", rays.I_vec)
        self.rotate_rays_local_basis(inverse_meridional=inverse_meridional)

    def get_intensity(self, scaling=1, scale_by_density=True):
        """
        calculate field intensity on surface for the 
        rays object scaled by photoselection
        """
        I = np.real(self.E_vec*np.conj(self.E_vec))*scaling#/
        self.I_per_dipole_xyz = np.mean(I, axis=0)
        I_per_dipole = np.sum(self.I_per_dipole_xyz, axis=1)
        I_sum = np.sum(I_per_dipole)
        # I tried scaling by "ray density" so the answers are independent
        # of ray sampling, but ray number/solid angle appears to vary slightly
        # so this is disabled for now
        if scale_by_density:
            ray_density = self.ray_density
        else:
            ray_density = 1
        self.I_vec = I
        self.I_total_norm = I_sum/ray_density
    
    def remove_escaped_rays(self, escaped=None):
        """is there a more efficient way of doing this?"""
        lost = deepcopy(self)
        if escaped is None:
            print(self.escaped)
            not_escaped = np.invert(self.escaped)
            print(not_escaped)

            escaped = self.escaped
        else:  # supply different escaped array that from self
            not_escaped = np.invert(escaped)
        self.E_vec = self.E_vec[:, not_escaped, :, :]
        self.k_vec = self.k_vec[not_escaped, : , :]
        self.transfer_matrix = self.transfer_matrix[:, not_escaped, :, :]
        self.phi = self.phi[not_escaped]
        self.theta = self.theta[not_escaped]
        self.rho = self.rho[not_escaped]
        self.area_scaling = self.area_scaling[not_escaped]
        print(self.areas)
        self.areas = self.areas[not_escaped]

        lost.E_vec = lost.E_vec[:, escaped, :, :]
        lost.k_vec = lost.k_vec[escaped, : , :]
        lost.transfer_matrix = lost.transfer_matrix[:, escaped, :, :]
        lost.phi = lost.phi[escaped]
        lost.theta = lost.theta[escaped]
        lost.rho = lost.rho[escaped]
        lost.area_scaling = lost.area_scaling[escaped]
        lost.areas = lost.areas[escaped]

        self.n_final = len(self.phi)
        return lost  # return part that is discarded

    
    def combine_rays(self, rays2):
        self.E_vec = np.append(self.E_vec, rays2.E_vec, axis=1)
        self.k_vec = np.append(self.k_vec, rays2.k_vec, axis=0)
        self.transfer_matrix = \
            np.append(self.transfer_matrix, self.transfer_matrix, axis=0)
        self.phi = np.append(self.phi, rays2.phi)
        self.theta = np.append(self.theta, rays2.theta)
        self.rho = np.append(self.rho, rays2.rho)
        self.area_scaling = np.append(self.area_scaling, rays2.area_scaling)
        self.areas = np.append(self.areas, rays2.areas)

    def set_zero_escaped_rays(self, escaped=None):
        if escaped is None:
            escaped = self.escaped
        self.E_vec[:, escaped, :, :] = 0
        # print("alternative minimum", self.alternative_minimum)

    def quiver_plot_debug(self, show_plots='2d', downsampling=50, n_rays=40,\
        E_vec_num=10, use_rho=False):
        started_merid = False
        if self.isMeridional:  # put back into non meridional basis
            self.meridional_transform(inverse=True)
            started_merid = True

        # do all
        E_vec_all_before = self.E_vec
        E_vec_all = self.transfer_matrix @ self.E_vec
        E_vec_all_after = E_vec_all
        tf_all = self.transfer_matrix
        E_vec_all = E_vec_all.reshape((E_vec_all.shape[0], E_vec_all.shape[1], 3))
        print("vec all shape x", E_vec_all[0, :, 0].shape)
        print("vec all shape y", E_vec_all[0, :, 1].shape)
        print("vec all shape conj y", np.conj(E_vec_all[0, :, 1]).shape)
        Ix = np.squeeze(E_vec_all[0, :, 0]*np.conj(E_vec_all[0, :, 0]))
        Iy = np.squeeze(E_vec_all[0, :, 1]*np.conj(E_vec_all[0, :, 1]))
        
        plt.figure()
        plt.scatter(E_vec_all[0, :, 0].real, E_vec_all[0, :, 1].real)
        plt.title("scatter for E real component all")
        plt.show()
        plt.figure()
        plt.scatter(E_vec_all[0, :, 0].imag, E_vec_all[0, :, 1].imag)
        plt.title("scatter for E imag component all")
        plt.show()
        plt.figure()
        plt.scatter(Ix, Iy)
        plt.title("scatter for E modulus component all")
        plt.show()


        downsample_mask = list(range(0,self.n,downsampling))
        if n_rays is not None:
            downsample_mask = np.random.randint(0, self.n-1, n_rays)
            downsample_mask = list(range(0,self.n,int(self.n/n_rays)))

        # E_vec_mask = E_vec_all[:, downsample_mask, :, :]    

        theta = self.theta[downsample_mask]
        phi = self.phi[downsample_mask]
        rho = self.rho[downsample_mask]

        z = np.cos(theta)
        y = np.sin(theta)*np.sin(phi)
        x = np.sin(theta)*np.cos(phi)
        print(np.all(np.abs(theta) < 1e-6) )
        if (np.mean(np.abs(theta)) < 1e-6)  or use_rho:# or any(rho > 1e-6):
            print("Plotting Rho")
            print(np.mean(np.abs(theta)) )
            z = np.zeros(len(theta))
            y = abs(rho)*np.sin(phi)
            x = abs(rho)*np.cos(phi)
            print("max x,y from rho", np.max(np.abs(x)), np.max(np.abs(y)))

        k_vec = self.k_vec[downsample_mask]
        transfer_matrix = self.transfer_matrix[:, downsample_mask, :, :]
        transfer_matrix = transfer_matrix.reshape((1,len(downsample_mask),3,3))
        escaped = np.array(self.escaped)
        escaped = escaped[downsample_mask]
        E_vec = self.E_vec[:, downsample_mask, :, :]
        not_escaped = np.invert(escaped)
        # not_escaped = np.ones_like(not_escaped)*True

        E_vec = E_vec[:, not_escaped, :, :]
        print("vec shape x", E_vec_all[0, :, 0].shape)

        k_vec = k_vec[not_escaped, : , :]
        transfer_matrix = transfer_matrix[:, not_escaped, :, :]
        phi = phi[not_escaped]
        theta = theta[not_escaped]
        x = x[not_escaped]
        y = y[not_escaped]
        z = z[not_escaped]
        # rho = rho[not_escaped]
        print("E_Vec before tf", E_vec.shape)
        print("tf resid", np.array_equal(tf_all , transfer_matrix))

        plt.figure()
        plt.scatter(E_vec[:,:,0,0], E_vec[:,:,1,0])
        plt.title("E before (E_vec)")
        plt.show()
        plt.figure()
        plt.scatter(E_vec_all_before[:,:,0,0]*np.conj(E_vec_all_before[:,:,0,0]), E_vec_all_before[:,:,1,0]*np.conj(E_vec_all_before[:,:,1,0]))
        plt.title("I before (E_vec_all)")
        plt.show()
        E_vec_before = E_vec
        E_vec = transfer_matrix @ E_vec  # this will be slow!
        print("E_Vec after tf", E_vec.shape)
        I_vec = np.sum(E_vec*E_vec, axis=0)
        I_vec = np.reshape(I_vec, (I_vec.shape[0], 3))
        k_vec = np.reshape(k_vec, (k_vec.shape[0], 3))
        I_vec_mag = np.sqrt(I_vec[:,0]**2 + I_vec[:,1]**2 + I_vec[:,2]**2 )
        I_vec_mag = I_vec_mag.reshape((I_vec_mag.shape[0], 1))
        print(I_vec_mag.shape)
        I_vec=I_vec/I_vec_mag

        print("E before resid", np.array_equal(E_vec_all_before, E_vec_before))
        print("E after resid", np.array_equal(E_vec,E_vec_all_after))

        print(I_vec.shape)
        print("max k_vec in x,y", np.max(np.abs(k_vec[:,0])), np.max(np.abs(k_vec[:,1])))
        print("max E_vec x, y", np.max(np.abs(E_vec[:, :, 0])), np.max(np.abs(E_vec[:, :, 1])))
        print("E shape", E_vec.shape)

        if E_vec_num is not None:
            ndipole = E_vec.shape[0]
            if ndipole > 1: # edge case for 1 dipole source
                ndipole -= 1
            E_vec_num_mask = list(range(0, ndipole, int(ceil(E_vec.shape[0]/E_vec_num))))
            E_vec = E_vec[E_vec_num_mask,:, :, :]
            print("E_vec.shape[1]", E_vec.shape[1])
        
        if show_plots == 'all':  # do 3d plot
            ax = plt.figure(figsize=[14,9]).add_subplot(projection='3d')
            #ax.quiver(x, y, z, I_vec[:, 0], I_vec[:, 1], I_vec[:, 2], length=0.12, normalize=True)
            ax.quiver(x, y, z, k_vec[:, 0], k_vec[:, 1], k_vec[:, 2], length=0.12, normalize=True, color='k')
            ax.quiver(x, y, z, E_vec[0,:, :, 0].real, E_vec[0,:, :, 1].real, E_vec[0,:, :, 2].real, scale=0.12, normalize=True, color='g')
            ax.quiver(x, y, z, E_vec[0,:, :, 0].imag, E_vec[0,:, :, 1].imag, E_vec[0,:, :, 2].imag, scale=0.12, normalize=True, color='purple')

            ax.set_xlim3d(-1, 1)
            ax.set_ylim3d(-1, 1)
            ax.set_zlim3d(-1, 1)
            plt.show()

        plt.figure()
        plt.scatter(E_vec_all_after[:, :, 0,0]*np.conj(E_vec_all_after[:, :, 0,0]), E_vec_all_after[:, :, 1,0]*np.conj(E_vec_all_after[:, :, 1,0]))
        plt.title("scatter for E modulus component, before reshape")
        plt.show()    
        print("E_vec.shape", E_vec.shape)
        print("E_vec.shape[1]", E_vec.shape[1])

        E_vec = E_vec.reshape((E_vec.shape[0], E_vec.shape[1], 3))
        k_vec = np.tile(k_vec, [E_vec.shape[0],1,1])
        x = np.tile(x, [E_vec.shape[0],1,1])
        y = np.tile(y, [E_vec.shape[0],1,1])
        z = np.tile(z, [E_vec.shape[0],1,1])
        # width = np.max([x,y,z])*0.005
        # E_width = width*0.9
        E_width = 0.005
        width = 0.005
        print(k_vec.shape)
        # print("Evec real", E_vec.real)
        figsc = plt.figure()
        plt.scatter(x,y)
        plt.show()

        fig = plt.figure(figsize=[10,3])
        ax = fig.add_subplot(131)

        ax.scatter(x,y)
        ax.quiver(x, y, E_vec[:, :, 0].real, E_vec[:, :, 1].real, color='g', width=E_width, scale=10)
        ax.quiver(x, y, E_vec[:, :, 0].imag, E_vec[:, :, 1].imag, color='purple', width=E_width, scale=10)
        ax.quiver(x, y, k_vec[:, :, 0], k_vec[:, :, 1], width=width, scale=10)
        ax.set_aspect('equal')
        ax1 = fig.add_subplot(132)
        ax1.scatter(x,z)
        ax1.quiver(x, z, E_vec[:, :, 0].real, E_vec[:, :, 2].real, color='g', width=E_width, scale=10)
        ax1.quiver(x, z, E_vec[:, :, 0].imag, E_vec[:, :, 2].imag, color='purple', width=E_width, scale=10)
        ax1.quiver(x, z, k_vec[:, :, 0], k_vec[:, :, 2], width=width, scale=10)
        ax1.set_aspect('equal')
        ax2 = fig.add_subplot(133)
        ax2.scatter(y,z)
        ax2.quiver(y, z, E_vec[:, :, 1].real, E_vec[:, :, 2].real, color='g', width=E_width, scale=10)
        ax2.quiver(y, z, E_vec[:, :, 1].imag, E_vec[:, :, 2].imag, color='purple', width=E_width, scale=10)
        ax2.quiver(y, z, k_vec[:, :, 1], k_vec[:, :, 2], width=width, scale=10)
        ax2.set_aspect('equal')
        plt.show()

        plt.figure()
        plt.scatter(E_vec_all_after[:, :, 0].real, E_vec_all_after[:, :, 1].real)
        plt.title("scatter for E real component all")
        plt.show()
        plt.figure()
        plt.scatter(E_vec_all_after[:, :, 0].imag, E_vec_all_after[:, :, 1].imag)
        plt.title("scatter for E imag component all")
        plt.show()
        plt.figure()
        plt.scatter(E_vec_all_after[:, :, 0]*np.conj(E_vec_all_after[:, :, 0]), E_vec_all_after[:, :, 1]*np.conj(E_vec_all_after[:, :, 1]))
        plt.title("scatter for E modulus component all")
        plt.show()

        print(E_vec_all_after[:, :, :,0].shape, E_vec.shape)
        plt.figure()
        plt.scatter(E_vec_all_after[:, :, :,0],E_vec)
        plt.show()

        plt.figure()
        plt.scatter(E_vec[:, :, 0].real, E_vec[:, :, 1].real)
        plt.title("scatter for E real component")
        plt.show()
        plt.figure()
        plt.scatter(E_vec[:, :, 0].imag, E_vec[:, :, 1].imag)
        plt.title("scatter for E imag component")
        plt.show()
        plt.figure()
        plt.scatter(E_vec[:, :, 0]*np.conj(E_vec[:, :, 0]), E_vec[:, :, 1]*np.conj(E_vec[:, :, 1]))
        plt.title("scatter for E modulus component")
        plt.show()

        if started_merid == True:
            self.meridional_transform(inverse=False)

        return self, E_vec
    
    def quiver_plot(self, show_plots='2d', downsampling=50, n_rays=40,\
        E_vec_num=10, use_rho=False):
        started_merid = False
        if self.isMeridional:  # put back into non meridional basis
            self.meridional_transform(inverse=True)
            started_merid = True

        # do all
        E_vec_all_before = self.E_vec
        E_vec_all = self.transfer_matrix @ self.E_vec
        tf_all = self.transfer_matrix
        E_vec_all = E_vec_all.reshape((E_vec_all.shape[0], E_vec_all.shape[1], 3))
        Ix = np.squeeze(E_vec_all[0, :, 0]*np.conj(E_vec_all[0, :, 0]))
        Iy = np.squeeze(E_vec_all[0, :, 1]*np.conj(E_vec_all[0, :, 1]))
        


        downsample_mask = list(range(0,self.n,downsampling))
        if n_rays is not None:
            downsample_mask = np.random.randint(0, self.n-1, n_rays)
            downsample_mask = list(range(0,self.n,int(self.n/n_rays)))

        # E_vec_mask = E_vec_all[:, downsample_mask, :, :]    

        theta = self.theta[downsample_mask]
        phi = self.phi[downsample_mask]
        rho = self.rho[downsample_mask]

        z = np.cos(theta)
        y = np.sin(theta)*np.sin(phi)
        x = np.sin(theta)*np.cos(phi)
        print(np.all(np.abs(theta) < 1e-6) )
        if (np.mean(np.abs(theta)) < 1e-6)  or use_rho:# or any(rho > 1e-6):
            print("Plotting Rho")
            print(np.mean(np.abs(theta)) )
            z = np.zeros(len(theta))
            y = abs(rho)*np.sin(phi)
            x = abs(rho)*np.cos(phi)
            print("max x,y from rho", np.max(np.abs(x)), np.max(np.abs(y)))

        k_vec = self.k_vec[downsample_mask]
        transfer_matrix = self.transfer_matrix[:, downsample_mask, :, :]
        transfer_matrix = transfer_matrix.reshape((1,len(downsample_mask),3,3))
        escaped = np.array(self.escaped)
        escaped = escaped[downsample_mask]
        E_vec = self.E_vec[:, downsample_mask, :, :]
        not_escaped = np.invert(escaped)
        # not_escaped = np.ones_like(not_escaped)*True

        E_vec = E_vec[:, not_escaped, :, :]

        k_vec = k_vec[not_escaped, : , :]
        transfer_matrix = transfer_matrix[:, not_escaped, :, :]
        phi = phi[not_escaped]
        theta = theta[not_escaped]
        x = x[not_escaped]
        y = y[not_escaped]
        z = z[not_escaped]
        # rho = rho[not_escaped]

        E_vec_before = E_vec
        E_vec = transfer_matrix @ E_vec  # this will be slow!
        I_vec = np.sum(E_vec*E_vec, axis=0)
        I_vec = np.reshape(I_vec, (I_vec.shape[0], 3))
        k_vec = np.reshape(k_vec, (k_vec.shape[0], 3))
        I_vec_mag = np.sqrt(I_vec[:,0]**2 + I_vec[:,1]**2 + I_vec[:,2]**2 )
        I_vec_mag = I_vec_mag.reshape((I_vec_mag.shape[0], 1))
        I_vec=I_vec/I_vec_mag


        if E_vec_num is not None:
            ndipole = E_vec.shape[0]
            if ndipole > 1: # edge case for 1 dipole source
                ndipole -= 1
            E_vec_num_mask = list(range(0, ndipole, int(ceil(E_vec.shape[0]/E_vec_num))))
            E_vec = E_vec[E_vec_num_mask,:, :, :]
            print("E_vec.shape[1]", E_vec.shape[1])
        
        if show_plots == 'all':  # do 3d plot
            ax = plt.figure(figsize=[14,9]).add_subplot(projection='3d')
            #ax.quiver(x, y, z, I_vec[:, 0], I_vec[:, 1], I_vec[:, 2], length=0.12, normalize=True)
            ax.quiver(x, y, z, k_vec[:, 0], k_vec[:, 1], k_vec[:, 2], length=0.12, normalize=True, color='k')
            ax.quiver(x, y, z, E_vec[0,:, :, 0].real, E_vec[0,:, :, 1].real, E_vec[0,:, :, 2].real, scale=0.12, normalize=True, color='g')
            ax.quiver(x, y, z, E_vec[0,:, :, 0].imag, E_vec[0,:, :, 1].imag, E_vec[0,:, :, 2].imag, scale=0.12, normalize=True, color='purple')

            ax.set_xlim3d(-1, 1)
            ax.set_ylim3d(-1, 1)
            ax.set_zlim3d(-1, 1)
            plt.show()

        E_vec = E_vec.reshape((E_vec.shape[0], E_vec.shape[1], 3))
        k_vec = np.tile(k_vec, [E_vec.shape[0],1,1])
        x = np.tile(x, [E_vec.shape[0],1,1])
        y = np.tile(y, [E_vec.shape[0],1,1])
        z = np.tile(z, [E_vec.shape[0],1,1])
        # width = np.max([x,y,z])*0.005
        # E_width = width*0.9
        E_width = 0.005
        width = 0.005
        # print("Evec real", E_vec.real)
        fig = plt.figure(figsize=[10,3])
        ax = fig.add_subplot(131)

        ax.scatter(x,y)
        ax.quiver(x, y, E_vec[:, :, 0].real, E_vec[:, :, 1].real, color='g', width=E_width, scale=10)
        ax.quiver(x, y, E_vec[:, :, 0].imag, E_vec[:, :, 1].imag, color='purple', width=E_width, scale=10)
        ax.quiver(x, y, k_vec[:, :, 0], k_vec[:, :, 1], width=width, scale=10)
        ax.set_aspect('equal')
        ax1 = fig.add_subplot(132)
        ax1.scatter(x,z)
        ax1.quiver(x, z, E_vec[:, :, 0].real, E_vec[:, :, 2].real, color='g', width=E_width, scale=10)
        ax1.quiver(x, z, E_vec[:, :, 0].imag, E_vec[:, :, 2].imag, color='purple', width=E_width, scale=10)
        ax1.quiver(x, z, k_vec[:, :, 0], k_vec[:, :, 2], width=width, scale=10)
        ax1.set_aspect('equal')
        ax2 = fig.add_subplot(133)
        ax2.scatter(y,z)
        ax2.quiver(y, z, E_vec[:, :, 1].real, E_vec[:, :, 2].real, color='g', width=E_width, scale=10)
        ax2.quiver(y, z, E_vec[:, :, 1].imag, E_vec[:, :, 2].imag, color='purple', width=E_width, scale=10)
        ax2.quiver(y, z, k_vec[:, :, 1], k_vec[:, :, 2], width=width, scale=10)
        ax2.set_aspect('equal')
        plt.show()

        if started_merid == True:
            self.meridional_transform(inverse=False)


