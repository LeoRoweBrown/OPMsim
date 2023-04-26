import numpy as np
from matplotlib import pyplot

class RayDiagram():
    def __init__(self, draw_options=None) -> None:
        pass
        self.draw_options = {
            'title': "Ray diagram"
        }
        for key in draw_options:
            self.draw_options[key] = draw_options[key]
    # for each ray history
    # calculate E field

    def get_fields_and_rays(self, rays, dipole_index):
        initial_E_vec = rays.history[0].E_vec[dipole_index,:,:,:]
        n_points = len(rays.history)
        all_k_vecs = np.empty((n_points, rays.history[0].k_vec.shape[0], 3))
        all_E_fields = np.empty((n_points, initial_E_vec.shape[0], 3))
        for n in range(n_points):
            all_k_vecs[n, :, :] = rays.history[n].k_vec
            transfer_mat = rays.history[n].transfer_matrix
            all_E_fields[n, :, :] = initial_E_vec @ transfer_mat