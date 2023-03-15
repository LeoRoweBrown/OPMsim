from copy import deepcopy
from matplotlib import pyplot as plt
import numpy as np
import warnings
import time

from . import detector
from . import optical_elements
from . import trace_system
from . import dipole_source
from . import anisotropy


class SystemOutput():
    def __init__(self, detector, pupil_plot) -> None:
        self.detector = detector
        self.pupil_plot = pupil_plot

def objective_system(
    lenses, source=None, title="Objective system", 
    ray_count=5000, show_plot_pupil=True, show_ray_plots=False,
    polariser=None, plot_options={}, vector_plots=None, entrace_pupil_flat=False):
    """
    O1, O2, O3 are dictionaries with fields NA, f, n and rotation
    source is a DipoleSource object
    """
    print("title param:",title)
    init_start = time.time()
    if source is None:
        source = dipole_source.DipoleSource(name='X-dipole')
        source.add_dipoles((0,0))

    # rays defined by collection NA of O1
    if len(lenses) == 0:
        O1 = {'NA': 0.95, 'n':1, 'f':0.003}
    else:
        O1 = lenses[0]
    sine_halfangle = O1['NA']/O1['n']
    source.get_rays_uniform_rings(sine_halfangle, O1['f'], ray_count)
    source.display_pupil_rays()

    elements = []

    for lens in lenses:
        rot = 0
        print(lens)
        if 'rotation' in lens:
            rot = lens['rotation']*np.pi/180
        print("Lens rotation:", rot)
        elements.append(optical_elements.SineLens(
            lens['NA'], lens['f'], n=lens['n'],
            yAxis_rotation=rot, show_plots=show_ray_plots))

    # add the polariser
    if polariser is not None:
        for n in range(np.size(polariser)):
            i = polariser['position']  # e.g. 2 (count from 0) means after O1
            psi = polariser['psi']
            elements.insert(i, optical_elements.LinearPolariser(psi))

    system = {'name': title, 'elements': elements}
    print(elements)

    if vector_plots is not None:
        system['vector_plots'] = vector_plots

    init_time = time.time() - init_start
    print("initialisation time in system %fs" % init_time)

    trace_start = time.time()
    detector, initial_detector = trace_system.trace_rays(
        system, source, entrace_pupil_flat=entrace_pupil_flat)
    trace_time = time.time() - trace_start
    print("time in trace_rays %fs" % trace_time)

    title += "\n n_dipoles=%d, n_rays=%d (initial n_rays=%d)" % (source.n_dipoles, detector.n_rays, detector.n_rays_initial)
    if show_plot_pupil:  
        # TODO: sort this stuff out... sometimes just refuses to plot initial for some cases
        # some recursive stuff going on wiht plot_options, ** issue?
        plot_options['title'] = title
        plot_options['caption'] = True  # plot details like Iy/Ix and RCE for first plot...
        print("Using this title....", plot_options['title'])
        detector.plot_pupil(**plot_options)  # plot pupil field
        plot_options['caption'] = False  # ...but no the second plot (O1)
        if entrace_pupil_flat:
            plot_options['title'] += str(" (intensity on flat, 2nd princ. surface of O1)")
        else:
            plot_options['title'] += str(" (intensity incident on O1)")
        initial_detector.plot_pupil(**plot_options)
        plot_options['title'] = ""

    return detector, initial_detector

def antisotropy_objective_system(lenses, source, title="Objective system", ray_count=5000):
    """
    O1, O2, O3 are dictionaries with fields NA, f, n and rotation
    source is a DipoleSource object
    """
    detector, initial_detector = objective_system(lenses, source, title, ray_count)
    r = anisotropy.calculate_anisotropy(detector)

    lens = lenses[0]
    collection_NA = optical_elements.SineLens(
            lens['NA'], lens['f'], n=lens['n']).sine_theta
    # TODO: remove this test
    collection_NA = lens['NA']/lens['n']

    if len(source.phi_d) > 1:
        ## anisotropy from Axelrod, only valid for x-excitation
        if source.excitation_polarisation != (0,0):
            warnings.warn("Dipole source has photoselection in non-x direction, theory will be invalid")
            r_theory = np.NaN
        else:
            r_theory = anisotropy.theoretical_anisotropy_population(collection_NA)
    else:
        r_theory = anisotropy.theoretical_anisotropy(collection_NA, (0,0))

    return (r, r_theory)


def anisotropy_test_no_obj(collection_NA, collection_f, bg=False, ray_count=5000):
    source = dipole_source.DipoleSource(name='x dipole')
    # source.generate_dipoles(1000)#
    excitation_polarisation = (0,0)
    source.add_dipoles((0,0))
    source.get_rays_uniform_rings(collection_NA, collection_f, ray_count=5000)

    elements_p = []
    # elements_p.append(optical_elements.LinearPolariser(excitation_polarisation[0]))

    system_p = {'name': 'Parallel arm anisotropy', 'elements': elements_p}

    source.get_rays_uniform_rings(collection_NA, collection_f, ray_count=5000)

    detector_p = trace_system.trace_rays(system_p, source)

    detector_p.plot_pupil(system_p['name'], fill_zeroes=bg)

    # reset rays/retrace rays
    source.get_rays_uniform_rings(collection_NA, collection_f, ray_count=5000)

    elements_s = []
    # elements_s.append(optical_elements.LinearPolariser(excitation_polarisation[0]+np.pi/2))

    system_s = {'name': 'Perpendicular arm anisotropy', 'elements': elements_s}
    detector_s = trace_system.trace_rays(system_s, source)

    detector_s.plot_pupil(system_p['name'], fill_zeroes=bg)

    r = anisotropy.calculate_anisotropy_rawdata(detector_p, detector_s, simulate_polariser=True)
    print("Anisotropy r =", r)

    r_theory = anisotropy.theoretical_anisotropy(collection_NA, excitation_polarisation)
    print("Theoretical anisotropy r =", r_theory)

    return (r, r_theory)

def anisotropy_measuring_system(collection_NA, collection_f, excitation_polarisation,
                                source=None, two_obj=False):
    """
    Measure anisoptropy of a 1 (or 2) objective system for given source and excitation
    polarisation. Default behaviour generates random dipoles and photoselects
    """
    
    init_start = time.time()
    if source is None:
        source = dipole_source.DipoleSource()
        source.generate_dipoles(500, method='uniform_phi_inbetween')
        excitation_polarisation = (0,0)
        source.classical_photoselection(excitation_polarisation)
    
    elements_p = []
    elements_p.append(optical_elements.SineLens(collection_NA, collection_f))
    print(excitation_polarisation)
    elements_p.append(optical_elements.LinearPolariser(excitation_polarisation[0]))
    if two_obj:
        # add a second lens to image over a curved surface, just to check it's the same
        elements_p.append(optical_elements.SineLens(collection_NA, collection_f))

    system_p = {'name': 'Parallel arm anisotropy', 'elements': elements_p}

    source.get_rays_uniform_rings(collection_NA, collection_f, ray_count=5000)

    init_time = time.time() - init_start
    print("initialisation time in anisotropy_measuring_system %fs" % init_time)

    trace_start = time.time()
    detector_p = trace_system.trace_rays(system_p, source)
    trace_time = time.time() - trace_start
    print("time in trace_rays %fs" % trace_time)

    detector_p.plot_pupil(system_p['name'])

    # reset rays/retrace rays
    source.get_rays_uniform_rings(collection_NA, collection_f, ray_count=5000)

    elements_s = []
    elements_s.append(optical_elements.SineLens(collection_NA, collection_f))
    elements_s.append(optical_elements.LinearPolariser(excitation_polarisation[0]+np.pi/2))
    if two_obj:
        elements_s.append(optical_elements.SineLens(collection_NA, collection_f))

    system_s = {'name': 'Perpendicular arm anisotropy', 'elements': elements_s}
    detector_s = trace_system.trace_rays(system_s, source)

    detector_s.plot_pupil(system_p['name'])

    r = anisotropy.calculate_anisotropy_rawdata(detector_p, detector_s)
    print("Anisotropy r =", r)

    if len(source.dipole_ensemble) > 1:
        ## anisotropy from Axelrod, only valid for x-excitation
        if dipole_source.excitation_polarisation != (0,0):
            warnings.warn("Dipole source has photoselection in non-x direction, theory will be invalid")
            r_theory = np.NaN
        else:
            r_theory = anisotropy.theoretical_anisotropy_population(collection_NA)
    else:
        r_theory = anisotropy.theoretical_anisotropy(collection_NA, excitation_polarisation)
    print("Theoretical anisotropy r =", r_theory)

    return (r, r_theory)
