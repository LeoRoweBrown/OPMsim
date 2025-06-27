from .base_element import Element
from .. import matrices

class LinearPolariser(Element):
    def __init__(self, psi, dz=0, label='', update_history=False):
        super().__init__(
            element_type='LinearPolariser',
            dz=dz,
            label=label)

        self.psi = psi
        self.update_history = update_history

    def trace_rays(self, rays, calculate_efield=False, debug_dir=None):
        if self.update_history:
            rays.update_history()

        # k-vector unaffected, only E-field, so just update transfer matrix
        polariser = matrices.optical_elements.polariser(self.psi)
        rays.transfer_matrix = polariser @ rays.transfer_matrix
