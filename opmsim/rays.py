from copy import deepcopy
# from tkinter import E
import numpy as np
import copy
from . import optical_matrices
import matplotlib.pyplot as plt

## vectorised version of ray

class PolarRays:
    def __init__(self,
                phi_array, theta_array, radius,
                area_array, lda=500e-9, keep_history=True):
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
        self.radius = radius

        self.isMeridional = False
        self.escaped = [False]*self.n
        self.areas = area_array
        self.area_scaling = np.ones(self.n)
        self.transfer_matrix = np.tile(np.identity(3), (1,self.n,1,1))
        # self.transfer_matrix = self.transfer_matrix.reshape(1,self.n,3,3)
        self.alternative_minimum = None  # used when E-values are set to 0 manually

        self.keep_history = keep_history
        self.ray_history = []

    def update_history(self, note=None):
        self.note = note
        if self.keep_history:
            self.ray_history.append(copy.deepcopy(self))

    def rotate_rays_local_basis(self, inverse_basis=False, inverse_meridional=True):
        """
        Rotate each E_vec to local frame (defined for ray i by phi[i] and theta[i])
        """
        # check orthogonality of E and k
        # print("dot product", np.sum(self.E_vec * self.k_vec, axis=1))
        # print("basis rotate transfer matrix before", self.transfer_matrix)
        # we want to start from non meridional basis, unless we are doing
        # inverse basis transform (inverse basis tf undoes meridional)
        if self.isMeridional and not inverse_basis:
            print("Inverting meridional")
            self.meridional_transform(inverse=True)
        basis_tensor = optical_matrices.rotate_basis_tensor(
            self.phi, self.theta, inverse_basis)
        # thinking about removing k_vec tf, but it's undone in merid so has to stay..
        self.k_vec = basis_tensor @ self.k_vec  
        self.transfer_matrix = basis_tensor @ self.transfer_matrix
        # print("basis rotate transfer matrix", self.transfer_matrix)

        self.isMeridional = True # basis rotation automatically does meridional tf
        if inverse_meridional:
            self.meridional_transform(inverse=True)  # recover global Ex and Ey


    def meridional_transform(self, inverse=False):
        if self.isMeridional != inverse:
            raise Exception("Cannot apply meridional transform twice in same direction")
        merid_tensor = optical_matrices.meridional_transform_tensor(
            self.phi, inverse)
        self.k_vec = merid_tensor @ self.k_vec
        self.isMeridional = not inverse
        self.transfer_matrix = merid_tensor @ self.transfer_matrix

    def finalize_rays_coordinates(self):

        if self.isMeridional:  # put back into non meridional basis
            self.meridional_transform(inverse=True)

        self.update_history()  # save rays before
        # print("before rotate", rays.I_vec)
        self.rotate_rays_local_basis()
    
    def remove_escaped_rays(self, escaped=None):
        """is there a more efficient way of doing this?"""
        lost = deepcopy(self)
        if escaped is None:
            not_escaped = np.invert(self.escaped)
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

    def quiver_plot(self, show_plots='2d', downsampling=50, n_rays=40,\
        E_vec_num=10, use_rho=False):
        started_merid = False
        if self.isMeridional:  # put back into non meridional basis
            self.meridional_transform(inverse=True)
            started_merid = True

        downsample_mask = list(range(0,self.n-1,downsampling))
        if n_rays is not None:
            downsample_mask = np.random.randint(0, self.n-1, n_rays)
            downsample_mask = list(range(0,self.n-1,int(self.n/n_rays)))

        theta = self.theta[downsample_mask]
        phi = self.phi[downsample_mask]
        rho = self.rho[downsample_mask]

        z = np.cos(theta)
        y = np.sin(theta)*np.sin(phi)
        x = np.sin(theta)*np.cos(phi)
        print(np.all(np.abs(theta) < 1e-6) )
        if (np.all(np.abs(theta) < 1e-6) ) or use_rho:# or any(rho > 1e-6):
            print("Plotting Rho")
            z = np.zeros(len(theta))
            y = abs(rho)*np.sin(phi)
            x = abs(rho)*np.cos(phi)


        k_vec = self.k_vec[downsample_mask]
        transfer_matrix = self.transfer_matrix[:, downsample_mask, :, :]
        transfer_matrix = transfer_matrix.reshape((1,len(downsample_mask),3,3))
        escaped = np.array(self.escaped)
        escaped = escaped[downsample_mask]
        E_vec = self.E_vec[:, downsample_mask, :, :]
        not_escaped = np.invert(escaped)

        E_vec = E_vec[:, not_escaped, :, :]
        k_vec = k_vec[not_escaped, : , :]
        transfer_matrix = transfer_matrix[:, not_escaped, :, :]
        phi = phi[not_escaped]
        theta = theta[not_escaped]
        x = x[not_escaped]
        y = y[not_escaped]
        z = z[not_escaped]
        # rho = rho[not_escaped]
        print()
        E_vec = transfer_matrix @ E_vec  # this will be slow!
        I_vec = np.sum(E_vec*E_vec, axis=0)
        I_vec = np.reshape(I_vec, (I_vec.shape[0], 3))
        k_vec = np.reshape(k_vec, (k_vec.shape[0], 3))
        I_vec_mag = np.sqrt(I_vec[:,0]**2 + I_vec[:,1]**2 + I_vec[:,2]**2 )
        I_vec_mag = I_vec_mag.reshape((I_vec_mag.shape[0], 1))
        print(I_vec_mag.shape)
        I_vec=I_vec/I_vec_mag

        print(I_vec.shape)
        print("max k_vec in x,y", np.max(np.abs(k_vec[:,0])), np.max(np.abs(k_vec[:,1])))
        if E_vec_num is not None:
            E_vec_num_mask = list(range(0, E_vec.shape[0]-1, int(E_vec.shape[0]/E_vec_num)))
            E_vec = E_vec[E_vec_num_mask, :, :]
        
        if show_plots == 'all':  # do 3d plot
            ax = plt.figure(figsize=[14,9]).add_subplot(projection='3d')
            #ax.quiver(x, y, z, I_vec[:, 0], I_vec[:, 1], I_vec[:, 2], length=0.12, normalize=True)
            ax.quiver(x, y, z, k_vec[:, 0], k_vec[:, 1], k_vec[:, 2], length=0.12, normalize=True, color='k')
            ax.quiver(x, y, z, E_vec[:, :, 0], E_vec[:, :, 1], E_vec[:, :, 2], length=0.12, normalize=True)
            ax.set_xlim3d(-1, 1)
            ax.set_ylim3d(-1, 1)
            ax.set_zlim3d(-1, 1)
            plt.show()
        E_vec = E_vec.reshape((E_vec.shape[0], E_vec.shape[1], 3))
        k_vec = np.tile(k_vec, [E_vec.shape[0],1,1])
        x = np.tile(x, [E_vec.shape[0],1,1])
        y = np.tile(y, [E_vec.shape[0],1,1])
        z = np.tile(z, [E_vec.shape[0],1,1])
        width = np.max([x,y,z])*0.005
        E_width = width*0.9
        print(k_vec.shape)
        fig = plt.figure(figsize=[10,3])
        ax = fig.add_subplot(131)
        ax.quiver(x, y, E_vec[:, :, 0], E_vec[:, :, 1], color='blue', width=E_width)
        ax.quiver(x, y, k_vec[:, :, 0], k_vec[:, :, 1], scale=13, width=width)
        ax.set_aspect('equal')
        ax1 = fig.add_subplot(132)
        ax1.quiver(x, z, E_vec[:, :, 0], E_vec[:, :, 2], color='blue', width=E_width)
        ax1.quiver(x, z, k_vec[:, :, 0], k_vec[:, :, 2], scale=13, width=width)
        ax1.set_aspect('equal')
        ax2 = fig.add_subplot(133)
        ax2.quiver(y, z, E_vec[:, :, 1], E_vec[:, :, 2], color='blue', width=E_width)
        ax2.quiver(y, z, k_vec[:, :, 1], k_vec[:, :, 2], scale=13, width=width)
        ax2.set_aspect('equal')
        plt.show()

        if started_merid == True:
            self.meridional_transform(inverse=False)
