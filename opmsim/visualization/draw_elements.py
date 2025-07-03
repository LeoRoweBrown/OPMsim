import numpy as np
from matplotlib import pyplot as plt
from matplotlib import axes
from ..optical_elements.sine_lens import SineLens
from ..optical_elements.base_element import Element
from ..matrices.transformation import rotate_y

def draw_sine_lens(ax: axes.Axes, sine_lens: SineLens, view='xz', color='k'):
    """
    Draw sine lnes on system plot.
    Note currently only xz view is supported, and rotation about y (from mirror or objective tilt),
    the local y values of the surfaces is just assumed to be 0, i.e. curve in xz plane, but in reality
    it is a spherical caps and a plane"""
    x, y, z = sine_lens.coords
    max_sine_theta = (sine_lens.NA / sine_lens.n)
    sine_thetas = np.linspace(-max_sine_theta, max_sine_theta, 50)
    phis = np.linspace(0, 2 * np.pi, 50)
    r = sine_lens.front_focal_length

    # maybe we should do the whole 3D, take a central cut or draw full mesh at some point? TODO
    # for t in sine_thetas:
    #     curved_x_3d = t * r * np.cos(phis)
    #     curved_z_3d = np.ones(len(phis)) * r * ((1 - t**2)**0.5)

    if sine_lens.flipped_orientation:
        curved_surface_x = sine_thetas * r
        curved_surface_z = r * (1 - (1 - sine_thetas**2)**0.5)
        flat_surface_x = np.array([-max_sine_theta, max_sine_theta]) * r
        flat_surface_z = np.array([0, 0])
    else:
        curved_surface_x = sine_thetas * r
        curved_surface_z = r * ((1 - sine_thetas**2)**0.5)
        flat_surface_x = np.array([-max_sine_theta, max_sine_theta]) * r
        flat_surface_z = np.array([r, r])

    # For now, we just assume y is zero because we only have y-rotation and only look in xz for now
    curved_surface_y = np.zeros_like(sine_thetas)
    flat_surface_y = np.zeros(2)

    curved_surface = np.array([
        curved_surface_x,
        curved_surface_y,
        curved_surface_z])
    flat_surface = np.array([
        flat_surface_x,
        flat_surface_y,
        flat_surface_z])

    # now convert into global basis?
    # get any y rotation on the lens. for lens needs to be negative
    rot_y = -sine_lens.y_axis_rotation
    curved_surface = sine_lens.basis @ rotate_y(rot_y) @ curved_surface
    flat_surface = sine_lens.basis @ rotate_y(rot_y) @ flat_surface
    print("Curved surface shape", curved_surface.shape)

    ax.plot(curved_surface[2, :] + z, curved_surface[0, :] + x, color=color)
    ax.plot(flat_surface[2, :] + z, flat_surface[0, :] + x, color=color)

def draw_line_element(ax: axes.Axes, element: Element, pupil_radius, rot_y=0, view='xz', color='k'):
    x, y, z = element.coords
    surface_z = np.array([0, 0])
    surface_y = np.array([0, 0])
    surface_x = np.array([-pupil_radius, pupil_radius])
    surface = np.array([
        surface_x,
        surface_y,
        surface_z])
    surface = element.basis @ rotate_y(rot_y) @ surface
    ax.plot(surface[2, :] + z, surface[0, :] + x, color=color)
