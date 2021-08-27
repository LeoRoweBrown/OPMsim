import numpy as np
import optical_elements

class OpticalSystem():
    """angles theta and phi are angle relative to last optical element"""
    def __init__(name="system1"):
        self.name = name
        self.element_list = []

    def add_thin_lens(dz, f, diameter, theta=0, phi=0, dx=0, dy=0):
        pass
        element = optical_elements.ThinLens(focal_length, diameter)
        element.coords = 
        
        element_list.append(element)

    def add_objective(magnification, NA):
        pass

    def add_tube_lens(dz, f, theta=0, phi=0):
        pass

    def add_plane_mirror(
        z, f, theta_surface, phi_surface, theta=0, phi=0, dx=0, dy=0):
        """
        I distinguish between theta and theta surface etc. because
        it would be annoying having to deal with the relative angle between
        the last element being changed after the tilted mirror in dOPM
        """
        pass

    def add_polariser(dz, diameter, angle, theta=0, phi=0, dx=0, dy=0):
        pass

    def add_waveplate(relative_phase):
        """pi/2 = quarter waveplate"""
        pass

    ### runtime methods
    def start_ray_tracing(self, rays):
        """ trace all rays """
        pass

    def trace_to_element(self, ray, index):
        """ trace single ray to next element, stop if ray lost """
        element = self.element_list[index]
        # ray [x, theta]

    def propagate(self, ray):
        """ propagate ray, may be redundantly small func """
        pass