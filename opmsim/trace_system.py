from copy import deepcopy
import numpy as np
from matplotlib import pyplot as plt
import time
from memory_profiler import profile
import warnings

from .detector import Detector
from . import dipole_source
from . import optical_elements
from .tools import graphics
from . import anisotropy


def count_lenses(elements):
    # number of lenses to determine if final wavefront is curved
    count = 0
    for i, element in enumerate(elements):
        if element.type == "SineLens":
            count += 1
    return count


# @profile
def trace_rays(elements, source, options):
    """
    trace the rays through the system and calculate the transfer matrix
    so that we can trace E vectors separately and efficiently at the
    end exploiting numpy C libraries

    parameters:
    elements (list<Element obj>): list of element objects (see optical_elements.py)
    source (DipoleSource obj): dipole_source object from dipole_source.py
    options (dict): options for ray tracing simulation
    """

    max_radius = None
    if count_lenses(elements) % 2 == 0:
        is_curved_pupil = True
        max_radius = source.NA
        print("Curved pupil")
    else:
        is_curved_pupil = False
        print("Flat pupil")

    # Curved affects how polar plot coordinates are derived -- i.e. for radius, from actual rho or sin(theta)
    detector = Detector(curved=is_curved_pupil)

    rays = source.rays

    if options["calculate_entrace_pupil"]:
        initial_rays = deepcopy(rays)

    for i, element in enumerate(elements):
        print("ELEMENT:", element)
        rays = element.trace_rays(rays)  # , keep_history=keep_history)
        # plot? 3d vector diagram
        if "vector_plots" in options:
            if type(options["vector_plots"]) is True:
                rays.quiver_plot()
            elif i in options["vector_plots"]:
                rays.quiver_plot()

    # apply inverse meridional transform or transforms
    # to coordinates on sphere if wavefront curved
    rays.finalize_rays_coordinates()  # inverse_meridional=False)

    # remove rays that are marked as escaped during the tracing
    if not options["keep_escaped"]:
        rays.remove_escaped_rays()

    # Apply transfer matrix to E-field vector
    rays.transfer_matrix = rays.transfer_matrix.reshape((1, rays.n_final, 3, 3))
    rays.e_field = rays.transfer_matrix @ rays.e_field

    source.emission_scaling = source.emission_scaling.reshape((rays.e_field.shape[0], 1, 1, 1))

    rays.get_intensity(source.emission_scaling)

    if source.emission_scaling.squeeze().size > 1:
        plt.figure()
        plt.hist(source.emission_scaling.squeeze())
        plt.xlabel("emission_scaling/photoselection scaling")
        plt.show()

    detector.final_energy = rays.total_intensity_normalized

    # for plotting with limiting NA and showing collection NA as boundary
    element = elements[0]
    detector.limiting_NA = element.NA
    detector.limiting_D = element.D
    if is_curved_pupil:
        detector.max_r = element.NA / element.n
        detector.max_r = element.NA / element.n
    else:
        detector.max_r = element.D / 2

    # maybe move this to detector? No harm done though
    print("Limiting radius for detector:", detector.max_r)
    print("Energy ratio (efficiency):", detector.final_energy / rays.initial_energy)
    print("Total energy per dipole per ray:", detector.final_energy)

    detector.relative_collection_efficiency = detector.final_energy / rays.initial_energy
    # equivalent to CE ideally
    detector.scaled_relative_collection_efficiency = rays.emission_efficiency * detector.relative_collection_efficiency
    detector.emission_efficiency = rays.emission_efficiency
    detector.collection_efficiency = detector.final_energy / rays.half_sphere_energy

    detector.n_dipoles = source.n_dipoles
    Energy_x = np.sum(rays.intensity_per_dipole_vector[:, 0])
    Energy_y = np.sum(rays.intensity_per_dipole_vector[:, 1])
    detector.Ix_Iy_ratio = Energy_x / Energy_y

    print("Energy from Ix", Energy_x)
    print("Energy from Iy", Energy_y)
    print("X/Y energy ratio =", detector.Ix_Iy_ratio)
    print("Half sphere energy", rays.half_sphere_energy)
    print("Initial energy", rays.initial_energy)
    print("half sphere energy NA", rays.average_energy_times_NA)

    # -----------------------------------------------

    # checking for cases where E not perpendicular with k (something went wrong)
    if any(np.abs(rays.intensity_per_dipole_vector[:, 2] > 1e-9)):  # check non-zero z comp
        print(((rays.total_intensity_normalized * rays.k_vec) ** 2).shape)
        dotp = np.sum((rays.total_intensity_normalized * rays.k_vec) ** 2, 1)
        print("dotp", dotp.shape)
        if any(dotp > 1e-9):
            print("Error in I dot product too!")
            print(dotp)
            print("max dot prod error", max(dotp))
        # print(rays.I_total)
        print("k_vec", rays.k_vec)
        print("mag of k_vec", np.sum(rays.k_vec * rays.k_vec, axis=1))
        print("e_field", rays.e_field)
        print(rays.transfer_matrix)
        warnings.warn("Iz is non zero in ray's frame!")
        # raise ValueError("Iz is non zero in ray's frame!")

    print("maxr", detector.max_r)
    detector.detect_rays(rays)  # calculate the pupil field

    # do same with initial rays to plot clipped pupil in O1 space
    # all this does is plot the initial field but only for rays that
    # are traced through the system

    if options["calculate_entrace_pupil"]:
        if options["entrance_pupil_flat"]:
            initial_detector = Detector(curved=False)
            initial_rays = elements[0].trace_rays(initial_rays)
        else:
            initial_detector = Detector(curved=True)
        initial_rays.finalize_rays_coordinates()  # does this for k_vec only

        # remove lost rays to reduce compute time on applying coord tf matrix
        lost_rays = initial_rays.remove_escaped_rays(rays.escaped)

        initial_rays.e_field = initial_rays.transfer_matrix @ initial_rays.e_field
        initial_rays.get_intensity(source.emission_scaling)

        initial_detector.detect_rays(initial_rays)
        initial_detector.n_dipoles = source.n_dipoles
        initial_detector.isinitial = True

        return detector, initial_detector
    return detector
