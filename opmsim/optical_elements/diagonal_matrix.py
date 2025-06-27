from .base_element import Element
from .. import matrices

class DiagonalMatrix(Element):
    def __init__(self, value, dz=0, label='', update_history=False):
        super().__init__(
            element_type='DiagonalMatrix',
            dz=dz,
            label=label)
        self.value = value  # diagonal value
        self.update_history = update_history

    def trace_rays(self, rays, calculate_efield=False, debug_dir=None):
        if self.update_history:
            rays.update_history()
        rays.transfer_matrix = \
            matrices.optical_elements.diagonal(self.value) @ rays.transfer_matrix
        return rays
