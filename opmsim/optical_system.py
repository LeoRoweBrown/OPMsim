from typing import Sequence, List, Union, cast
from warnings import warn
import datetime
import os
import numpy as np
from .optical_elements.base_element import Element
from .optical_elements.sine_lens import SineLens
from .detector import Detector
from .rays import PolarRays
from .dipole_source import DipoleSource
from copy import copy, deepcopy

class OpticalSystem():
    """High-level class representing optical system"""
    def __init__(self,
                 elements: Sequence[Element] = (),  # sequence instead of list for covariant types
                 source: DipoleSource = DipoleSource((0, 0))):
        """
        Args:
            elements (List[BaseElement]): list of optical element objects, see optical_elements folder
            source (DipoleSource): dipole source object comprising electric dipoles, generates rays,
                stores E-field, etc. Defaults to DipoleSource((0, 0)), a single x-dipole source.
        """
        self.name = "default_optical_system"
        self.elements: List[Element] = list(elements)  # for mutability internally
        self.source = source
        self.NA = 1
        self.initial_half_angle = np.pi / 2  # entrance pupil half angle
        self.ray_count = 7500  # should this be set in source instead?
        self.preview_ray_count = 100
        self.calculate_efield_sequentially = False
        self.detector = None
        self.pupil_plot = None
        self.auto_orient_lenses = True

        # current date for debug file
        date = datetime.datetime.now()
        date_string = f"{date.year}{date.month}{date.day}"
        time_string = f"{date.hour}{date.minute}{date.second}"
        self.debug_dir = f"./debug/{self.name}/{date_string}/{time_string}"
        self.save_debug_data = False

    def add_element(self, element: Element, idx=None, dz=0):
        if idx is None:
            idx = len(self.elements)
        self.elements.insert(idx, element)

    def remove_element(self, idx=-1):
        self.elements.pop(idx)

    def trace_system(self, compute_efield=True, plot_pupil=True, preview=False):
        """Call to trace rays: to calculate final ray k-vector and e-vector (if compute_efield is true)

        Args:
            source (DipoleSource, optional):
                DipoleSource object, . Defaults to DipoleSource((0, 0)), a single x-dipole.
            compute_efield (bool, optional): _description_. Defaults to True.
        """

        # get initial ray cone at primary objective
        if len(self.elements) > 0:
            primary_objective = cast(SineLens, self.elements[0])
            ffl = primary_objective.front_focal_length
            NA, n = primary_objective.NA, primary_objective.n
        else:  # for testing
            ffl = 1
            NA, n = 0.9, 1
        half_angle = np.arcsin(NA / n)
        debug_dir = self.debug_dir if self.save_debug_data else ""

        if self.auto_orient_lenses:
            self.orient_lenses()  # orient the SineLenses i.e., flip every other one
            self.update_element_positions()

        # Generates ray cone and draws it to first lens (by a distance = ffl)
        ray_count = self.preview_ray_count if preview else self.ray_count
        self.source.get_rays_uniform(half_angle, ffl, ray_count)
        rays: PolarRays = self.source.rays

        n_elements = len(self.elements)

        print(f"Element list: {self.elements} ({n_elements} elements)")

        for n, element in enumerate(self.elements):
            print(f"Tracing {element} ({n + 1}/{n_elements}) label: {element.label}")
            # pass rays object to trace_rays to mutate/trace them
            rays.propagate(element.dz)
            element.trace_rays(rays)
            if self.calculate_efield_sequentially:  # update e-field if doing sequentially
                element.update_efield(rays)
            if self.save_debug_data:  # save matrix data
                subdir = f"n{n:.02d}"
                subdir = subdir if element.label == "" else f"{subdir}_{element.label}"
                fulldir = os.path.join(self.debug_dir, subdir)
                rays.save_debug_data(fulldir, save_efield=self.calculate_efield_sequentially)

        if preview:  # if doing ray preview, don't need to do any E-field calculation
            return

        if compute_efield and not self.calculate_efield_sequentially:
            rays.e_field = rays.transfer_matrix @ rays.e_field

        # Final check for k.E = 0
        rays.verify_dot_product()

        if self.is_exit_pupil_wavefront_flat():
            print("Final wavefront is curved!")
        else:
            print("Final wavefront is flat!")
        self.detector = Detector(self.is_exit_pupil_wavefront_flat())
        self.detector.detect_rays(rays)

        if plot_pupil:
            self.pupil_plot = self.detector.plot_exit_pupil()

    def preview_rays(self):
        self.trace_system(compute_efield=False)

    def orient_lenses(self, detect_new_basis=True):
        """autoflip the lenses to be correct and get basis right"""
        flip = False
        n_lenses = 0
        print("Orientations:")
        for i, element in enumerate(self.elements):
            # if basis changes,
            if i > 0 and element.use_previous_basis:
                element.basis = self.elements[i - 1].basis
            if type(element) is SineLens:
                if n_lenses % 2 == 1:
                    element.flipped_orientation = True
                elif element.dz > 0:
                    warn("Non-flipped lens given non-zero displacement along optical axis (dz)."
                         "Reverting to dz=0, which corresponds to collimating lenses distance f from images/objects")
                    element.dz = 0
                print(f"Lens {i}: flipped={element.flipped_orientation}")
                flip = not flip  # every other element is fliped
                n_lenses = n_lenses + 1

    def update_element_positions(self):
        current_coords = np.array([0, 0, 0])
        for i, element in enumerate(self.elements):
            print("current coords", current_coords)
            optical_axis = element.basis[2, :]
            element.coords = current_coords + element.dz * optical_axis
            current_coords = element.coords + element.thickness * optical_axis

    def is_exit_pupil_wavefront_flat(self):
        n_lenses = 0
        for element in self.elements:
            if type(element) is SineLens:
                n_lenses += 1
        return (n_lenses % 2) == 0
