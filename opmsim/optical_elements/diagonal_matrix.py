from .base_element import Element
from .. import matrices

class DiagonalMatrix(Element):
    def __init__(self, value, update_history=False):
        self.type = 'DiagonalMatrix'
        self.value = value  #
        self.update_history = update_history

    def trace_rays(self, rays, calculate_efield=False, debug_dir=None):
        if self.update_history:
            rays.update_history()
        if rays.isMeridional:  # put back into non meridional basis
            rays.meridional_transform(inverse=True)
        rays.transfer_matrix = \
            matrices.optical_elements.diagonal(self.value) @ rays.transfer_matrix
        return rays
