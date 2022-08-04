from operator import not_
import numpy as np
import copy
import optical_matrices

## vectorised version of ray

class PolarRays:
    def __init__(self,
                phi_array, theta_array, I_array, k_array,
                area_array, rho_array=None, lda=500e-9, keep_history=True):
        """
        Arguments:
        lda <float> -- wavelength of ray

        Methods:
        """
        # print(phi_array)
        self.n = len(phi_array)
        self.lda = lda  # wavelength
        # these are all updated/affected by transforms
        # pad to work with parallel matrix mul in the form of (3,3,N) tensors
        self.I_vec = np.reshape(I_array, (self.n, 3, 1))
        self.k_vec = np.reshape(k_array, (self.n, 3, 1))  # n by 3 matrix (x, y, z)
        self.phi = phi_array
        self.theta_in = theta_array  # NOT UPDATED
        self.theta = theta_array
        if rho_array is None:
            self.rho = np.zeros_like(phi_array)
        else:
            self.rho = rho_array  # allow to be negative for correct refraction
        self.isMeridional = False
        self.escaped = [False]*self.n
        self.areas = area_array
        self.area_scaling = np.ones(self.n)

        self.keep_history = keep_history
        self.ray_history = []

    def update_history(self, note=None):
        self.note = note
        if self.keep_history:
            self.ray_history.append(copy.deepcopy(self))

    def rotate_rays_local_basis(self, inverse=False):
        """
        Rotate each I_vec to local frame (defined for ray i by phi[i] and theta[i])
        """
        """
        # we don't actually use complex numbers anymore
        # rotate into E_z = 0 and square then put back into global basis
        basis_tensor = optical_matrices.rotate_basis_tensor(phi_list, theta_list)
        E_vec = basis_tensor @ np.reshape(E_vec, (n_rays,3,1))
        meridional = optical_matrices.meridional_transform_tensor(phi_list) # recover global xy
        invert_meridional = optical_matrices.meridional_transform_tensor(phi_list, inverse=True)
        E_vec = invert_meridional @ E_vec
        I = E_vec * np.absolute(E_vec)  # we add intensities (add field incoherently)
        I = meridional @ I  # back into meridional 
        # print("Iz after rotation", I[:,2])
        invert_basis_tensor = np.linalg.inv(basis_tensor)
        """
        # check orthogonality of I and k
        print("dot product", np.sum(self.I_vec * self.k_vec, axis=1))
        if self.isMeridional:
            print("Inverting meridional")
            self.meridional_transform(inverse=True)
        basis_tensor = optical_matrices.rotate_basis_tensor(self.phi, self.theta)
        self.I_vec = basis_tensor @ self.I_vec
        self.k_vec = basis_tensor @ self.k_vec
        Ix, Iy, Iz = self.I_vec[:,0], self.I_vec[:,1], self.I_vec[:,2]
        # if |Iz|^2 > 0.001*|I|^2 (and Iz > 0) something has (definitely) gone wrong
        if any((Iz**2 > 1e-3*(Ix**2 + Iy**2 + Iz**2)) * (Iz > 1e-9)):  # the end bit is a bit fudgy
            print(Iz)
            print(self.k_vec)
            raise Exception("E_z is not zero in ray's frame for one or more rays!")

        self.isMeridional = True # basis rotation automatically does meridional tf
        self.meridional_transform(inverse=True)  # recover global Ix and Iy

    def meridional_transform(self, inverse=False):
        if self.isMeridional != inverse:
            raise Exception("Cannot apply meridional transform twice in same direction")
        merid_tensor = optical_matrices.meridional_transform_tensor(
            self.phi, inverse)
        self.I_vec = merid_tensor @ self.I_vec
        self.k_vec = merid_tensor @ self.k_vec
        self.isMeridional = not inverse

    def remove_escaped_rays(self):
        """is there a more efficient way of doing this"""
        not_escaped = np.invert(self.escaped)
        self.I_vec = self.I_vec[not_escaped, : , :]
        self.k_vec = self.k_vec[not_escaped, : , :]
        self.phi = self.phi[not_escaped]
        self.theta = self.theta[not_escaped]
        self.rho = self.rho[not_escaped]
        self.area_scaling = self.area_scaling[not_escaped]
        self.areas = self.areas[not_escaped]
        self.n_final = len(self.phi)