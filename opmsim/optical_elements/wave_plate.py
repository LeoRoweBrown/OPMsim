from .base_element import Element
from .. import matrices

class WavePlate(Element):
    def __init__(self, psi, delta, update_history=False, plot_debug=False):
        self.type = 'WavePlate'
        self.psi = psi  # angle of fast axis from x axis
        self.delta = delta  # amount of retardation e.g delta=pi/2 for qwp
        self.update_history = update_history
        self.plot_debug = plot_debug

    def trace_rays(self, rays, update_history=False):
        if self.update_history:
            rays.update_history()

        rays.transfer_matrix = \
            matrices.optical_elements.wave_plate(self.psi, self.delta) @ rays.transfer_matrix
