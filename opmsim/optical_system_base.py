from copy import deepcopy
from matplotlib import pyplot as plt
import numpy as np
import warnings
import time

from . import optical_elements
from . import trace_system
from . import dipole_source
from . import anisotropy


class OpticalSystem():
    def __init__(self, title, elements, source=None, options={}):
        """
        options:
            'ray_count': (int) number of rays to trace
            'vector_plot': (list) indices of elements for which to plot ray diagrams
            'show_input_rays': (bool) whether to display figure of rays from source
            'calculate_entrance_pupil' (bool) whether to plot initial field in O1 space
            'entrace_pupil_flat' (bool) if previous is true, whether to plot curved or flat pupil of O1
        """
        init_start = time.time()
        self.title = title
        self.elements = elements
        self.source = source
        self.options = {
            'ray_count': 5000,
            'vector_plot': range(len(elements)),  # plot vectors at every element
            'show_input_rays': False,
            'calculate_entrace_pupil': True,
            'entrace_pupil_flat': False  # for tracing to first surface, to show field on O1
        }
        for key in options:
            self.options[key] = options[key]

        ## set up optical system

        if self.source is None:
            self.source = dipole_source.DipoleSource(name='Uniformally distributed dipoles')
            self.source.generate_dipoles(1000)  # uniform generation
            # source.classical_photoselection((0,0))  # simulate excitation

        # rays defined by collection NA of O1
        O1 = elements[0]
        if O1.type != 'SineLens':
            raise TypeError("First element must be a sine lens!")

        source.get_rays_uniform_rings(O1.sine_theta, O1.focal_length, self.options['ray_count'])
        if self.options['show_input_rays']:
            source.display_pupil_rays()

        init_time = time.time() - init_start
        print("initialisation time in system %fs" % init_time)
    
    def trace(self):
        trace_start = time.time()
        self.detector, self.initial_detector = trace_system.trace_rays(
            self.elements, self.source, self.options)
        trace_time = time.time() - trace_start
        print("time in trace_rays %fs" % trace_time)

    def _plot_pupil(self, plot_options={}, initial_pupil=False):
        default_plot_title = \
            "Pupil intensity plot"
        sim_details = "n_dipoles=%d, n_rays=%d (initial n_rays=%d)"\
                % (self.source.n_dipoles, self.detector.n_rays,
                    self.detector.n_rays_initial)

        self.plot_options = {
            'title': default_plot_title,
            'caption': True,
            'add_sim_details': True
        }
        for key in plot_options:
            self.plot_options[key] = plot_options[key]

        if initial_pupil:
            self.plot_options['title'] += " -- intensity incident on O1"
        if self.plot_options['add_sim_details']:
            self.plot_options['title'] += "\n" + sim_details
        # only used on 'runtime' to edit the title, but this dicitonary is
        # passed as kwargs to plot_pupil... messy I know
        self.plot_options.pop('add_sim_details', None)

        pupil = self.detector.plot_pupil(**self.plot_options)  # plot pupil field

        return pupil

    def plot_pupil(self, plot_options={}):
        self._plot_pupil(plot_options)

    def plot_initial_pupil(self, plot_options_initial={}):
        self._plot_pupil(plot_options_initial, initial_pupil=True)
