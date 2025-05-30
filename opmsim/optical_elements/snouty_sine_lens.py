def compute_transmission(self, rays):
    """yet to implement wavelength dependence"""

    n0 = self.n

    if self.interface_ris is None and self.ar_coating_ri is None:
        return

    # get incident angles
    if rays.isMeridional:
        rays.meridional_transform(inverse=True)
    n_vec = np.array([0, 0, -1])  # flat tip, i.e. snouty
    n_vec = n_vec / np.linalg.norm(n_vec)
    n_vec = n_vec.reshape(1, 3, 1)

    p = np.cross(rays.k_vec, n_vec, axis=1)  # get p vector (kxN) (s wave comp)
    k_dot_normal = np.sum(rays.k_vec * n_vec, 1)
    np.abs(k_dot_normal.squeeze())

    r = np.cross(rays.k_vec, p, axis=1)  # get r vector (kxp) (p wave comp)

    # normalize since we compute the angles without the normalization factor...
    p = self.normalize(p)
    r = self.normalize(r)

    parallel = r[:, :, 0]
    senkrecht = p[:, :, 0]

    # compute angles
    k_x_n = np.cross(rays.k_vec, n_vec, axis=1)
    sin_mr_1theta = np.linalg.norm(abs(k_x_n), axis=1)
    theta_i = np.arcsin(sin_mr_1theta)

    # compare to actual theta..
    plt.plot(theta_i, label="theta_i")
    plt.plot(rays.theta, label="rays.theta")
    plt.legend()
    plt.show()

    plt.hist(theta_i, label="theta_i")
    plt.legend()
    plt.show()
    plt.hist(rays.theta, label="rays.theta")
    plt.legend()
    plt.show()

    # print(e_after.shape)
    # print(rays.e_field.shape)
    # print("ray diff", rays_after-rays.k_vec)
    # print("e diff", e_after-e_before)
    if self.fresnel_debug_savedir is not None:
        mat_t_p = mat_t[:, 0, 0]
        mat_t_s = mat_t[:, 1, 1]
        np.savetxt(os.path.join(self.fresnel_debug_savedir, "theta.csv"), theta_i)
        np.savetxt(os.path.join(self.fresnel_debug_savedir, "M_fresnel.csv"), mat_t.reshape(mat_t.shape[0], 9))
        np.savetxt(os.path.join(self.fresnel_debug_savedir, "mat_t_p.csv"), mat_t_p)
        np.savetxt(os.path.join(self.fresnel_debug_savedir, "mat_t_s.csv"), mat_t_s)

        mat_r = optical_matrices.fresnel_matrix(
            theta_i, n0, self.interface_ris[0], reflection=True)
        mat_r_p = mat_r[:, 0, 0]
        mat_r_s = mat_r[:, 1, 1]
        np.savetxt(os.path.join(self.fresnel_debug_savedir, "T_p.csv"), 1 - np.real(mat_r_p * mat_r_p))
        np.savetxt(os.path.join(self.fresnel_debug_savedir, "T_s.csv"), 1 - np.real(mat_r_s * mat_r_s))

    rays.transfer_matrix = (ps_project_inv @ mat_t @ ps_project @ rays.transfer_matrix)  # .reshape(shape_before)
    # print(np.shape(rays.transfer_matrix))
