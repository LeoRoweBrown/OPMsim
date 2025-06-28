"""
def quiver_plot_debug(
            self, show_plots='2d', downsampling=50, n_rays=40,
            e_field_num=10, use_rho=False):
        started_merid = False
        if self.is_meridional:  # put back into non meridional basis
            self.meridional_transform(inverse=True)
            started_merid = True

        # do all
        e_field_all_before = self.e_field
        e_field_all = self.transfer_matrix @ self.e_field
        e_field_all_after = e_field_all
        tf_all = self.transfer_matrix
        e_field_all = e_field_all.reshape((e_field_all.shape[0], e_field_all.shape[1], 3))
        print("vec all shape x", e_field_all[0, :, 0].shape)
        print("vec all shape y", e_field_all[0, :, 1].shape)
        print("vec all shape conj y", np.conj(e_field_all[0, :, 1]).shape)
        Ix = np.squeeze(e_field_all[0, :, 0] * np.conj(e_field_all[0, :, 0]))
        Iy = np.squeeze(e_field_all[0, :, 1] * np.conj(e_field_all[0, :, 1]))

        # Make the plot
        plt.figure()
        plt.scatter(e_field_all[0, :, 0].real, e_field_all[0, :, 1].real)
        plt.title("scatter for E real component all")
        plt.show()
        plt.figure()
        plt.scatter(e_field_all[0, :, 0].imag, e_field_all[0, :, 1].imag)
        plt.title("scatter for E imag component all")
        plt.show()
        plt.figure()
        plt.scatter(Ix, Iy)
        plt.title("scatter for E modulus component all")
        plt.show()

        # don't plot all the rays, it would be too crowded
        downsample_mask = list(range(0, self.n, downsampling))
        if n_rays is not None:
            downsample_mask = np.random.randint(0, self.n - 1, n_rays)
            downsample_mask = list(range(0, self.n, int(self.n / n_rays)))

        theta = self.theta[downsample_mask]
        phi = self.phi[downsample_mask]
        rho = self.rho[downsample_mask]

        z = np.cos(theta)
        y = np.sin(theta) * np.sin(phi)
        x = np.sin(theta) * np.cos(phi)
        print(np.all(np.abs(theta) < 1e-6))
        if (np.mean(np.abs(theta)) < 1e-6) or use_rho:  # or any(rho > 1e-6):
            print("Plotting Rho")
            print(np.mean(np.abs(theta)))
            z = np.zeros(len(theta))
            y = abs(rho) * np.sin(phi)
            x = abs(rho) * np.cos(phi)
            print("max x,y from rho", np.max(np.abs(x)), np.max(np.abs(y)))

        k_vec = self.k_vec[downsample_mask]
        transfer_matrix = self.transfer_matrix[:, downsample_mask, :, :]
        transfer_matrix = transfer_matrix.reshape((1, len(downsample_mask), 3, 3))
        escaped = np.array(self.escaped)
        escaped = escaped[downsample_mask]
        e_field = self.e_field[:, downsample_mask, :, :]
        not_escaped = np.invert(escaped)
        # not_escaped = np.ones_like(not_escaped)*True

        e_field = e_field[:, not_escaped, :, :]
        print("vec shape x", e_field_all[0, :, 0].shape)

        k_vec = k_vec[not_escaped, :, :]
        transfer_matrix = transfer_matrix[:, not_escaped, :, :]
        phi = phi[not_escaped]
        theta = theta[not_escaped]
        x = x[not_escaped]
        y = y[not_escaped]
        z = z[not_escaped]
        # rho = rho[not_escaped]
        print("e_field before tf", e_field.shape)
        print("tf resid", np.array_equal(tf_all, transfer_matrix))

        plt.figure()
        plt.scatter(e_field[:, :, 0, 0], e_field[:, :, 1, 0])
        plt.title("E before (e_field)")
        plt.show()
        plt.figure()
        plt.scatter(e_field_all_before[:, :, 0, 0] * np.conj(e_field_all_before[:, :, 0, 0]),
                    e_field_all_before[:, :, 1, 0] * np.conj(e_field_all_before[:, :, 1, 0]))
        plt.title("I before (e_field_all)")
        plt.show()
        e_field_before = e_field
        e_field = transfer_matrix @ e_field  # this will be slow!
        print("e_field after tf", e_field.shape)
        intensity_vector = np.sum(e_field * e_field, axis=0)
        intensity_vector = np.reshape(intensity_vector, (intensity_vector.shape[0], 3))
        k_vec = np.reshape(k_vec, (k_vec.shape[0], 3))
        intensity_vector_mag = np.sqrt(intensity_vector[:, 0]**2 + intensity_vector[:, 1]**2 + intensity_vector[:, 2] ** 2)
        intensity_vector_mag = intensity_vector_mag.reshape((intensity_vector_mag.shape[0], 1))
        print(intensity_vector_mag.shape)
        intensity_vector = intensity_vector / intensity_vector_mag

        print("E before resid", np.array_equal(e_field_all_before, e_field_before))
        print("E after resid", np.array_equal(e_field, e_field_all_after))

        print(intensity_vector.shape)
        print("max k_vec in x,y", np.max(np.abs(k_vec[:, 0])), np.max(np.abs(k_vec[:, 1])))
        print("max e_field x, y", np.max(np.abs(e_field[:, :, 0])), np.max(np.abs(e_field[:, :, 1])))
        print("E shape", e_field.shape)

        if e_field_num is not None:
            ndipole = e_field.shape[0]
            if ndipole > 1:  # edge case for 1 dipole source
                ndipole -= 1
            e_field_num_mask = list(range(0, ndipole, int(ceil(e_field.shape[0]/e_field_num))))
            e_field = e_field[e_field_num_mask, :, :, :]
            print("e_field.shape[1]", e_field.shape[1])

        if show_plots == 'all':  # do 3d plot
            ax = plt.figure(figsize=[14, 9]).add_subplot(projection='3d')
            ax.quiver(x, y, z, k_vec[:, 0], k_vec[:, 1], k_vec[:, 2], length=0.12, normalize=True, color='k')
            ax.quiver(x, y, z, e_field[0,:, :, 0].real, e_field[0, :, :, 1].real, e_field[0, :, :, 2].real,
                      scale=0.12, normalize=True, color='g')
            ax.quiver(x, y, z, e_field[0,:, :, 0].imag, e_field[0, :, :, 1].imag, e_field[0, :, :, 2].imag,
                      scale=0.12, normalize=True, color='purple')

            ax.set_xlim3d(-1, 1)
            ax.set_ylim3d(-1, 1)
            ax.set_zlim3d(-1, 1)
            plt.show()

        plt.figure()
        plt.scatter(e_field_all_after[:, :, 0, 0] * np.conj(e_field_all_after[:, :, 0, 0]), 
                    e_field_all_after[:, :, 1, 0] * np.conj(e_field_all_after[:, :, 1, 0]))
        plt.title("scatter for E modulus component, before reshape")
        plt.show()    
        print("e_field.shape", e_field.shape)
        print("e_field.shape[1]", e_field.shape[1])

        e_field = e_field.reshape((e_field.shape[0], e_field.shape[1], 3))
        k_vec = np.tile(k_vec, [e_field.shape[0], 1, 1])
        x = np.tile(x, [e_field.shape[0], 1, 1])
        y = np.tile(y, [e_field.shape[0], 1, 1])
        z = np.tile(z, [e_field.shape[0], 1, 1])
        # width = np.max([x,y,z])*0.005
        # E_width = width*0.9
        E_width = 0.005
        width = 0.005
        print(k_vec.shape)
        # print("Evec real", e_field.real)
        figsc = plt.figure()
        plt.scatter(x, y)
        plt.show()

        fig = plt.figure(figsize=[10, 3])
        ax = fig.add_subplot(131)

        ax.scatter(x, y)
        ax.quiver(x, y, e_field[:, :, 0].real, e_field[:, :, 1].real, color='g', width=E_width, scale=10)
        ax.quiver(x, y, e_field[:, :, 0].imag, e_field[:, :, 1].imag, color='purple', width=E_width, scale=10)
        ax.quiver(x, y, k_vec[:, :, 0], k_vec[:, :, 1], width=width, scale=10)
        ax.set_aspect('equal')
        ax1 = fig.add_subplot(132)
        ax1.scatter(x, z)
        ax1.quiver(x, z, e_field[:, :, 0].real, e_field[:, :, 2].real, color='g', width=E_width, scale=10)
        ax1.quiver(x, z, e_field[:, :, 0].imag, e_field[:, :, 2].imag, color='purple', width=E_width, scale=10)
        ax1.quiver(x, z, k_vec[:, :, 0], k_vec[:, :, 2], width=width, scale=10)
        ax1.set_aspect('equal')
        ax2 = fig.add_subplot(133)
        ax2.scatter(y, z)
        ax2.quiver(y, z, e_field[:, :, 1].real, e_field[:, :, 2].real, color='g', width=E_width, scale=10)
        ax2.quiver(y, z, e_field[:, :, 1].imag, e_field[:, :, 2].imag, color='purple', width=E_width, scale=10)
        ax2.quiver(y, z, k_vec[:, :, 1], k_vec[:, :, 2], width=width, scale=10)
        ax2.set_aspect('equal')
        plt.show()

        plt.figure()
        plt.scatter(e_field_all_after[:, :, 0].real, e_field_all_after[:, :, 1].real)
        plt.title("scatter for E real component all")
        plt.show()
        plt.figure()
        plt.scatter(e_field_all_after[:, :, 0].imag, e_field_all_after[:, :, 1].imag)
        plt.title("scatter for E imag component all")
        plt.show()
        plt.figure()
        plt.scatter(e_field_all_after[:, :, 0] * np.conj(e_field_all_after[:, :, 0]),
                    e_field_all_after[:, :, 1] * np.conj(e_field_all_after[:, :, 1]))
        plt.title("scatter for E modulus component all")
        plt.show()

        print(e_field_all_after[:, :, :, 0].shape, e_field.shape)
        plt.figure()
        plt.scatter(e_field_all_after[:, :, :, 0], e_field)
        plt.show()

        plt.figure()
        plt.scatter(e_field[:, :, 0].real, e_field[:, :, 1].real)
        plt.title("scatter for E real component")
        plt.show()
        plt.figure()
        plt.scatter(e_field[:, :, 0].imag, e_field[:, :, 1].imag)
        plt.title("scatter for E imag component")
        plt.show()
        plt.figure()
        plt.scatter(e_field[:, :, 0] * np.conj(e_field[:, :, 0]), e_field[:, :, 1] * np.conj(e_field[:, :, 1]))
        plt.title("scatter for E modulus component")
        plt.show()

        if started_merid:
            self.meridional_transform(inverse=False)

        return self, e_field

    def quiver_plot(
            self, show_plots='2d', downsampling=50, n_rays=40,
            e_field_num=10, use_rho=False):
        started_merid = False
        if self.is_meridional:  # put back into non meridional basis
            self.meridional_transform(inverse=True)
            started_merid = True

        # do all
        e_field_all_before = self.e_field
        e_field_all = self.transfer_matrix @ self.e_field
        tf_all = self.transfer_matrix
        e_field_all = e_field_all.reshape((e_field_all.shape[0], e_field_all.shape[1], 3))
        Ix = np.squeeze(e_field_all[0, :, 0] * np.conj(e_field_all[0, :, 0]))
        Iy = np.squeeze(e_field_all[0, :, 1] * np.conj(e_field_all[0, :, 1]))

        downsample_mask = list(range(0, self.n, downsampling))
        if n_rays is not None:
            downsample_mask = np.random.randint(0, self.n - 1, n_rays)
            downsample_mask = list(range(0, self.n, int(self.n / n_rays)))

        theta = self.theta[downsample_mask]
        phi = self.phi[downsample_mask]
        rho = self.rho[downsample_mask]

        z = np.cos(theta)
        y = np.sin(theta) * np.sin(phi)
        x = np.sin(theta) * np.cos(phi)
        print(np.all(np.abs(theta) < 1e-6))
        if (np.mean(np.abs(theta)) < 1e-6) or use_rho:  # or any(rho > 1e-6):
            print("Plotting Rho")
            print(np.mean(np.abs(theta)))
            z = np.zeros(len(theta))
            y = abs(rho) * np.sin(phi)
            x = abs(rho) * np.cos(phi)
            print("max x,y from rho", np.max(np.abs(x)), np.max(np.abs(y)))

        k_vec = self.k_vec[downsample_mask]
        transfer_matrix = self.transfer_matrix[:, downsample_mask, :, :]
        transfer_matrix = transfer_matrix.reshape((1, len(downsample_mask), 3, 3))
        escaped = np.array(self.escaped)
        escaped = escaped[downsample_mask]
        e_field = self.e_field[:, downsample_mask, :, :]
        not_escaped = np.invert(escaped)

        e_field = e_field[:, not_escaped, :, :]

        k_vec = k_vec[not_escaped, : , :]
        transfer_matrix = transfer_matrix[:, not_escaped, :, :]
        phi = phi[not_escaped]
        theta = theta[not_escaped]
        x = x[not_escaped]
        y = y[not_escaped]
        z = z[not_escaped]
        # rho = rho[not_escaped]

        e_field_before = e_field
        e_field = transfer_matrix @ e_field  # this will be slow!
        intensity_vector = np.sum(e_field * e_field, axis=0)
        intensity_vector = np.reshape(intensity_vector, (intensity_vector.shape[0], 3))
        k_vec = np.reshape(k_vec, (k_vec.shape[0], 3))
        intensity_vector_mag = np.sqrt(intensity_vector[:, 0]**2 + intensity_vector[:, 1]**2 + intensity_vector[:, 2] ** 2)
        intensity_vector_mag = intensity_vector_mag.reshape((intensity_vector_mag.shape[0], 1))
        intensity_vector = intensity_vector / intensity_vector_mag

        if e_field_num is not None:
            ndipole = e_field.shape[0]
            if ndipole > 1:  # edge case for 1 dipole source
                ndipole -= 1
            e_field_num_mask = list(range(0, ndipole, int(ceil(e_field.shape[0] / e_field_num))))
            e_field = e_field[e_field_num_mask, :, :, :]
            print("e_field.shape[1]", e_field.shape[1])

        if show_plots == 'all':  # do 3d plot
            ax = plt.figure(figsize=[14, 9]).add_subplot(projection='3d')
            ax.quiver(x, y, z, k_vec[:, 0], k_vec[:, 1], k_vec[:, 2], length=0.12, normalize=True, color='k')
            ax.quiver(x, y, z, e_field[0, :, :, 0].real, e_field[0, :, :, 1].real, e_field[0, :, :, 2].real,
                      scale=0.12, normalize=True, color='g')
            ax.quiver(x, y, z, e_field[0, :, :, 0].imag, e_field[0, :, :, 1].imag, e_field[0, :, :, 2].imag,
                      scale=0.12, normalize=True, color='purple')

            ax.set_xlim3d(-1, 1)
            ax.set_ylim3d(-1, 1)
            ax.set_zlim3d(-1, 1)
            plt.show()

        e_field = e_field.reshape((e_field.shape[0], e_field.shape[1], 3))
        k_vec = np.tile(k_vec, [e_field.shape[0],1,1])
        x = np.tile(x, [e_field.shape[0],1,1])
        y = np.tile(y, [e_field.shape[0],1,1])
        z = np.tile(z, [e_field.shape[0],1,1])
        # width = np.max([x,y,z])*0.005
        # E_width = width*0.9
        E_width = 0.005
        width = 0.005
        # print("Evec real", e_field.real)
        fig = plt.figure(figsize=[10,3])
        ax = fig.add_subplot(131)

        ax.scatter(x, y)
        ax.quiver(x, y, e_field[:, :, 0].real, e_field[:, :, 1].real, color='g', width=E_width, scale=10)
        ax.quiver(x, y, e_field[:, :, 0].imag, e_field[:, :, 1].imag, color='purple', width=E_width, scale=10)
        ax.quiver(x, y, k_vec[:, :, 0], k_vec[:, :, 1], width=width, scale=10)
        ax.set_aspect('equal')
        ax1 = fig.add_subplot(132)
        ax1.scatter(x, z)
        ax1.quiver(x, z, e_field[:, :, 0].real, e_field[:, :, 2].real, color='g', width=E_width, scale=10)
        ax1.quiver(x, z, e_field[:, :, 0].imag, e_field[:, :, 2].imag, color='purple', width=E_width, scale=10)
        ax1.quiver(x, z, k_vec[:, :, 0], k_vec[:, :, 2], width=width, scale=10)
        ax1.set_aspect('equal')
        ax2 = fig.add_subplot(133)
        ax2.scatter(y, z)
        ax2.quiver(y, z, e_field[:, :, 1].real, e_field[:, :, 2].real, color='g', width=E_width, scale=10)
        ax2.quiver(y, z, e_field[:, :, 1].imag, e_field[:, :, 2].imag, color='purple', width=E_width, scale=10)
        ax2.quiver(y, z, k_vec[:, :, 1], k_vec[:, :, 2], width=width, scale=10)
        ax2.set_aspect('equal')
        plt.show()

        if started_merid:
            self.meridional_transform(inverse=False)

"""