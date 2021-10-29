import numpy as np

# Base class for all elements
# I include the possibiliy of off-axis elements and rotated about x even though
# this is unlikely to be supported (maybe rotations will)

class Element():
    def __init__(self, focal_length, diameter, dz=0, dx=0, dy=0, theta=0, phi=0):
        self.coords(dz, dx, dy, theta, phi)

    def coords(self, dz=0, dx=0, dy=0, theta=0, phi=0):  # why did I make this a func
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.theta = theta
        self.phi = phi
        
    def trace_ray(self, ray):
        """ once ray is traced to the surface of the element run this """
        pass

class SineLens(Element):
    """Ideal lens that meets Abbe sine condition"""
    def __init__(self, focal_length, diameter, dz=0):
        self.type = 'SineLens'
        self.focal_length = focal_length
        self.diameter = diameter
        self.theta = 0
        self.phi = 0
        # set position of element
        super(SineLens, self).__init__(
            focal_length, diameter, dz, dx=0, dy=0, theta=0, phi=0)

class IdealFlatMirror(Element):
    """
    Ideal flat mirror with rotation about x axis
    May try making mirror with different p and s reflectance but requires
    decomposition of these components and it's within 1% difference anyway.
    """
    def __init__(self, focal_length, diameter, dz=0, rot_x=0):
        self.type = 'IdealFlatMirror'
        self.reflectance = 1
        self.diameter = diameter
        self.theta = 0
        self.phi = 0
        # set position of element
        super(IdealFlatMirror, self).__init__(
            focal_length, diameter, dz, dx=0, dy=0, theta=0, phi=0)

        ## TODO work out how to find intersection and angle between mirror normal and ray
        # maybe change coord system then back as soon as reflection is done?
    def _trace_to_surface(self, ray):
        ray.theta

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