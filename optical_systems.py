
from copy import deepcopy
import detector
import optical_elements
import trace_system
import dipole_source
from matplotlib import pyplot as plt
import numpy as np
import warnings
import anisotropy
import time

class SystemOutput():
    def __init__(self, detector, pupil_plot) -> None:
        self.detector = detector
        self.pupil_plot = pupil_plot

def objective_system(lenses, source=None, title="Objective system"):
    """
    O1, O2, O3 are dictionaries with fields NA, f, n and rotation
    source is a DipoleSource object
    """

    init_start = time.time()
    if source is None:
        source = dipole_source.DipoleSource(name='X-dipole')
        source.add_dipoles(0,0)

    # rays defined by collection NA of O1
    O1 = lenses[0]
    sine_halfangle = O1['NA']/O1['n']
    source.get_rays_uniform_rings(sine_halfangle, O1['f'], ray_count=5000)
    source.display_pupil_rays()
    source.view_pupil()

    elements = []

    for lens in lenses:
        rot = 0
        print(lens)
        if 'rotation' in lens:
            rot = lens['rotation']*np.pi/180
        print("Lens rotation:", rot)
        elements.append(optical_elements.SineLens(
            lens['NA'], lens['f'], n=lens['n'], xAxis_rotation=rot))

    system = {'name': title, 'elements': elements}

    init_time = time.time() - init_start
    print("initialisation time in system %fs" % init_time)

    trace_start = time.time()
    detector = trace_system.trace_rays(system, source)
    trace_time = time.time() - trace_start
    print("time in trace_rays %fs" % trace_time)

    detector.plot_pupil(title, fill_zeroes=False)  # plot pupil field

    return detector


def anisotropy_test_no_obj(collection_NA, collection_f):
    source = dipole_source.DipoleSource(name='x dipole')
    # source.generate_dipoles(1000)#
    excitation_polarisation = (0,0)
    source.add_dipoles(0,0)
    source.get_rays_uniform_rings(collection_NA, collection_f, ray_count=5000)

    elements_p = []
    # elements_p.append(optical_elements.LinearPolariser(excitation_polarisation[0]))

    system_p = {'name': 'Parallel arm anisotropy', 'elements': elements_p}

    source.get_rays_uniform_rings(collection_NA, collection_f, ray_count=5000)

    detector_p = trace_system.trace_rays(system_p, source)

    detector_p.plot_pupil(system_p['name'])

    # reset rays/retrace rays
    source.get_rays_uniform_rings(collection_NA, collection_f, ray_count=5000)

    elements_s = []
    # elements_s.append(optical_elements.LinearPolariser(excitation_polarisation[0]+np.pi/2))

    system_s = {'name': 'Perpendicular arm anisotropy', 'elements': elements_s}
    detector_s = trace_system.trace_rays(system_s, source)

    detector_s.plot_pupil(system_p['name'])

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
