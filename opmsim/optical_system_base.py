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
        title (string): description of simulation (not used in plotting)
        elements (list<Element obj>): list of element objects (see optical_elements.py)
        source (DipoleSource obj): dipole_source object from dipole_source.py

        options (dict):
            'ray_count': (int) number of rays to trace
            'vector_plot': (list or bool) indices of elements for which to plot ray 
                diagrams, or if true, make quiver plots after each element,
            'show_input_rays': (bool) whether to display figure of rays from source
            'calculate_entrance_pupil' (bool) whether to plot initial field in O1 space
            'entrance_pupil_flat' (bool) if previous is true, whether to plot curved or flat pupil of O1
                    calculate_entrance_pupil (bool): make another detector object for the 
                                        initial rays at the entrance pupil
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
            'entrance_pupil_flat': False,  # for tracing to first surface, to show field on O1
            'plot_ray_sphere': False,
            'draw_rays': False,
            'max_rays_stored': 200,
            'ray_dist': "uniform",
            'custom_rays': False,
            'keep_escaped': False
        }
        for key in options:
            self.options[key] = options[key]

        ## set up optical system

        if self.source is None:
            self.source = dipole_source.DipoleSource(name='Uniformally distributed dipoles')
            self.source.generate_dipole_ensemble(1000)  # uniform generation
            # source.classical_photoselection((0,0))  # simulate excitation

        # rays defined by collection NA of O1
        O1 = elements[0]
        if O1.type != 'SineLens':
            raise TypeError("First element must be a sine lens!")
        if self.options['draw_rays']:
            for n in range(len(elements)):
                elements[n].update_history = True

        if not self.options['custom_rays']:
            source.get_rays_uniform(O1.sine_theta, O1.focal_length,
                                        self.options['ray_count'], 
                                        ray_dist=self.options['ray_dist'],
                                        plot_sphere=self.options['plot_ray_sphere'])
        
        
        if self.options['show_input_rays']:
            source.display_pupil_rays()
        source.rays.num_rays_saved = self.options['max_rays_stored']

        init_time = time.time() - init_start
        print("initialisation time in system %fs" % init_time)

    
    def trace(self):
        trace_start = time.time()
        self.detector, self.initial_detector = trace_system.trace_rays(
            self.elements, self.source, self.options)
        trace_time = time.time() - trace_start
        print("time in trace_rays %fs" % trace_time)
        if self.options['draw_rays']:
            self.draw_rays()

    def _plot_pupil(self, plot_options={}, initial_pupil=False):
        default_plot_title = \
            "Pupil intensity plot"
        sim_details = "n_dipoles=%d, n_rays=%d (initial n_rays=%d)"\
                % (self.source.n_dipoles, self.detector.n_rays,
                    self.detector.n_rays_initial)

        self.plot_options = {
            'title': default_plot_title,
            'caption': True,
            'add_sim_details': True,
            'pupil_boundary_radius': None
        }
        for key in plot_options:
            self.plot_options[key] = plot_options[key]

        if initial_pupil:
            self.plot_options['title'] += " -- intensity incident on O1"
        # only used on 'runtime' to edit the title, but this dicitonary is
        # passed as kwargs to plot_pupil... messy I know

        pupil = self.detector.plot_pupil(**self.plot_options)  # plot pupil field

        return pupil

    def plot_pupil(self, plot_options={}):
        self._plot_pupil(plot_options)

    def plot_initial_pupil(self, plot_options_initial={}):
        self._plot_pupil(plot_options_initial, initial_pupil=True)

    def draw_rays(self, dipole=0, ray='all'):
        # option for sum of dipoles as I field?
        if ray == 'all':
            ray_mask = np.round(np.linspace(0, self.detector.n_rays, self.options['max_rays_stored']))
        elif hasattr(ray, "__len__"):
            ray_mask = ray
        elif isinstance(ray, int):
            ray_mask = ray
        else:
            raise ValueError("Invalid input for ray, supply array of ray indices, a single index or 'all'")
        ray_mask = np.array(set(ray_mask)) # remove duplicates

        pass