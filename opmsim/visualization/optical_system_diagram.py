import numpy as np
from typing import Sequence, Union
from matplotlib import pyplot as plt
from opmsim.visualization import draw_elements
from opmsim.optical_system import OpticalSystem
from opmsim.optical_elements import Element, SineLens, FlatMirror, WavePlate, LinearPolariser
from opmsim.rays import PolarRays

class OpticalSystemDiagram:
    def __init__(self, system: OpticalSystem) -> None:
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot()
        self.ax.set_aspect('equal')
        self.max_pupil_radius = 0
        self.drawn_element_objs = []
        self.ray_plot_refs = []
        self.element_plot_refs = []
        self.plot_only_in_plane = True  # TODO not really used right now
        self.optical_system = system
        self.estimate_max_pupil_height()

    def set_optical_system(self, system: OpticalSystem):
        """Setter to make clearer the intent to update the optical system that is plotted"""
        self.optical_system = system
        self.estimate_max_pupil_height()  # update pupil height calculation

    def estimate_max_pupil_height(self):
        max_pupil_diameter = 0
        for element in self.optical_system.elements:
            if isinstance(element, SineLens):
                max_pupil_diameter = max(element.D, max_pupil_diameter)
        self.max_pupil_radius = max_pupil_diameter / 2

    def clear_rays(self):
        for plot in self.ray_plot_refs:
            plot.remove()  # remove the plot from the axis
        self.ray_plot_refs = []  # reset the list of plot references

    def draw_rays(self):
        self.clear_rays()
        ray_paths = self.optical_system.source.rays.path_coords
        ray_paths_local = self.optical_system.source.rays.path_coords_local

        # Global positions to plot
        x, y, z = ray_paths[:, 0, :], ray_paths[:, 1, :], ray_paths[:, 2, :]  # shape: N_rays x 3 x N_traces
        x, y, z = x.squeeze(), y.squeeze(), z.squeeze()

        # And local positions (relative to optical axis) maybe can aid in only plotting rays in xz plane?
        xl, yl, zl = ray_paths_local[:, 0, :], ray_paths_local[:, 1, :], ray_paths_local[:, 2, :]
        xl, yl, zl = x.squeeze(), y.squeeze(), z.squeeze()

        ray_plot = self.ax.plot(z.T, x.T, linestyle='-', marker='')
        self.ray_plot_refs += ray_plot  # plot returns list, add to concat
        self.draw_elements()  # redraw the elements on top

    def clear_elements(self):
        for plot in self.element_plot_refs:
            print(plot)
            plot.remove()  # remove the plot from the axis
        self.element_plot_refs = []  # reset the list of plot references

    def draw_elements(self):
        self.clear_elements()
        for element in self.optical_system.elements:
            self.draw_element(element)

    def draw_element(self, element, **kwargs):
        print("drawing", element)
        if isinstance(element, SineLens):
            print("plotting", element, "as SineLens")
            plot_ = draw_elements.draw_sine_lens(self.ax, element)
        elif isinstance(element, FlatMirror):
            print("plotting", element, "as FlatMirror")
            plot_ = draw_elements.draw_line_element(
                self.ax, element, rot_y=element.rot_y, pupil_radius=self.max_pupil_radius)
        else:
            print("plotting", element, "as (flat) Element")
            plot_ = draw_elements.draw_line_element(self.ax, element, pupil_radius=self.max_pupil_radius)
        self.element_plot_refs += plot_  # the plot_ is returned as a list, so add not append
