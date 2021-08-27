import numpy as np

# Base class for all elements

class Element():
    def __init__(self, focal_length, diameter, dz=0, dx=0, dy=0, theta=0, phi=0):
        self.coords(dz, dx, dy, theta, phi)

    def coords(self, dz=0, dx=0, dy=0, theta=0, phi=0):
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.theta = theta
        self.phi = phi
        
    def trace_ray(self, ray):
        """ once ray is traced to the surface of the element run this """
        pass

class ThinLens(Element):
    def __init__(self, focal_length, diameter, dz=0, dx=0, dy=0, theta=0, phi=0):
        self.type = 'ThinLens'
        self.focal_length = focal_length
        self.diameter = diameter
        self.theta = theta
        self.phi = phi
        # set position of element
        super(ThinLens, self).__init__(
            focal_length, diameter, dz, dx, dy, theta, phi)
    
    def trace_ray(self, ray):
        """ once ray is traced to the surface of the element run this """
        # RTM
        vector2 = np.matmul(ray.vector)
         

class TubeLens(ThinLens):
    """ A bit redundant when ThinLens exists """
    def __init__(self, focal_length, diameter, dz=0):
        self.type = 'TubeLens'
        self.focal_length = focal_length
        self.diameter = diameter
        # set position of element
        super(TubeLens, self).__init__(
            focal_length, diameter, dz)
    
    def trace_ray(self, ray):
        """ once ray is traced to the surface of the element run this """
        pass