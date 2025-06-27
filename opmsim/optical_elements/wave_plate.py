from .base_element import Element
from .. import matrices

class WavePlate(Element):
    def __init__(self, psi, delta, dz=0, label='', update_history=False, plot_debug=False):
        super().__init__(
            element_type='WavePlate',
            dz=dz,
            label=label
        )
        self.type = 'WavePlate'
        self.psi = psi  # angle of fast axis from x axis
        self.delta = delta  # amount of retardation e.g delta=pi/2 for qwp
        self.update_history = update_history
        self.plot_debug = plot_debug

    def trace_rays(self, rays, calculate_efield=False, debug_dir=None):
        if self.update_history:
            rays.update_history()

        rays.transfer_matrix = \
            matrices.optical_elements.wave_plate(self.psi, self.delta) @ rays.transfer_matrix
