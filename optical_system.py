import numpy as np
import optical_elements

class OpticalSystem():
    """
    Angles angle_x and angle_y are angle relative to last optical element: e.g.
    If there is a mirror to reflect the beam by 90 degrees, the element
    after the beam has angle_y (probably angle_y) = 90 degrees
    """
    def __init__(self, name="system1"):
        self.name = name
        self.element_list = []

    def get_absolute_positions(self):
        els = self.element_list
        origin_pos = [0, 0, 0]  # x y z order
        current_pos = origin_pos
        origin_angle = [0, 0]  # angle_x, angle_y
        current_angle = origin_angle
        self.element.coords_absolute = []
        for n in len(els):
            el = els[n]
            dx = el['dx']
            dy = el['dy']
            dz = el['dz']
            angle_y = el['angle_y']*np.pi/180
            angle_x = el['angle_x']*np.pi/180
            dx_absolute = dx * np.cos(angle_y) - dz * np.sin(angle_y)
            dy_absolute = dx * np.sin(angle_x)*np.sin(angle_y) + dy * np.cos(angle_x)\
                + dz * np.sin(angle_x)*np.cos(angle_y)
            dz_absolute = dx * np.cos(angle_x)*np.sin(angle_y) - dy * np.sin(angle_x)\
                + dz * np.cos(angle_x)*np.cos(angle_y)

            current_angle[1] = (current_angle[1] + el['angle_y']) % (360.0)  # degrees
            current_angle[0] = (current_angle[0] + el['angle_x']) % (360.0)
            current_pos = [dx_absolute, dy_absolute, dz_absolute] + current_pos

            self.element.coords_absolute.append(
                {'dx': current_pos[0], 'dy': current_pos[1], 'dz': current_pos[2],
                 'angle_x': current_angle[0], 'angle_y': current_angle[1]}
            )
            self.elememt.

    def add_dipole_source(self):
        pass

    def add_thin_lens(self, dz, f, diameter, angle_y=0, angle_x=0, dx=0, dy=0):
        """angles supplied as degrees, we work in radians"""
        angle_x = angle_x*180/np.pi
        angle_y = angle_y*180/np.pi
        element = optical_elements.ThinLens(f, diameter)
        element.coords = {'dz': dz, 'dx': dx, 'dy': dy, 'angle_x':angle_x, 'angle_y':angle_y}

        self.element_list.append(element)

    def add_objective(self, magnification, NA,  angle_y=0, angle_x=0, dx=0, dy=0):
        angle_x = angle_x*180/np.pi
        angle_y = angle_y*180/np.pi
        pass

    def add_tube_lens(self, dz, f):
        pass

    def add_plane_mirror(self,
        z, f, angle_x_surface, angle_y_surface, angle_x=0, angle_y=0, dx=0, dy=0):
        """
        I distinguish between angle_x and angle_x surface etc. because
        it would be annoying having to deal with the relative angle between
        the last element being changed after the tilted mirror in dOPM
        """
        angle_x = angle_x*180/np.pi
        angle_y = angle_y*180/np.pi
        pass

    def add_polariser(self, dz, diameter, angle, angle_x=0, angle_y=0, dx=0, dy=0):
        angle_x = angle_x*180/np.pi
        angle_y = angle_y*180/np.pi
        pass

    def add_waveplate(self, relative_phase, angle_x=0, angle_y=0):
        angle_x = angle_x*180/np.pi
        angle_y = angle_y*180/np.pi
        """pi/2 = quarter waveplate"""
        pass

    ### runtime methods
    def start_ray_tracing(self, rays):
        """ trace all rays """
        pass

    def trace_to_element(self, ray, index):
        """ trace single ray to next element, stop if ray lost """
        element = self.element_list[index]
        # ray [x, angle_x]

    def propagate(self, ray):
        """ propagate ray, may be redundantly small func because of trace_to_element """
        pass