import numpy as np
from numpy.core.fromnumeric import trace
from detector import Detector, combine_detectors
import dipole_source
import optical_elements
import detector as det
import binning_detector as bin_det
import graphics
from matplotlib import pyplot as plt
from multiprocessing import Pool, process, current_process
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import time
import coord_tf

class SystemOutput():
    def __init__(self, detector, pupil_plot) -> None:
        self.detector = detector
        self.pupil_plot = pupil_plot

def trace_single_ray(elements, ray):

    for i, element in enumerate(elements):
        #element.apply_matrix(ray)
        ray = element.apply_matrix(ray)
    return ray

def trace_dipole_rays_mp(elements, detector, dipole):

    for n in range(len(dipole.ray_list)):
        for i, element in enumerate(elements):
            dipole.ray_list[n] = element.apply_matrix(dipole.ray_list[n])
    detector._add_field_fast(dipole)  # calculate the pupil field
    return detector

def trace_dipole_rays_mp_test(elements, dipole):

    for n in range(len(dipole.ray_list)):
        ray = dipole.ray_list[n]
        for i, element in enumerate(elements):
            dipole.ray_list[n] = element.apply_matrix(ray)
        coord_tf.finalize_coord_system(ray)
        dipole.ray_list[n] = ray
    
    return dipole

def trace_dipole_rays_mp_notf(elements, dipole):
    for n in range(len(dipole.ray_list)):
        ray = dipole.ray_list[n]
        for i, element in enumerate(elements):
            dipole.ray_list[n] = element.apply_matrix(ray)
        dipole.ray_list[n] = ray

    return dipole

def count_lenses(elements):
    count = 0
    for i, element in enumerate(elements):
        if element.type == 'SineLens':
            count += 1
    return count

def contains_element(elements, name):
    count = 0
    for i, element in enumerate(elements):
        if element.type == name:
            count += 1
    return count

def get_first_and_last_lens(elements):
    idxs = []
    for i, element in enumerate(elements):
        if element.type == 'SineLens':
            idxs.append(i)
    elements[min(idxs)].lens_position = 'first'
    elements[max(idxs)].lens_position = 'last'
    return elements

def trace_rays(system, source, collection_NA, method2=False, legacy=False):
    elements = system['elements']
    # decide whether to use ray radius (height) or theta angle
    max_radius=None
    if count_lenses(elements) % 2 == 0:
        is_curved_pupil = True
        max_radius=collection_NA
        print("Curved pupil")
    else:
        is_curved_pupil = False
        print("Flat pupil")

    dipole_count = len(source.dipole_ensemble)
    detector = det.Detector(curved=is_curved_pupil, max_radius=max_radius)

    for n in range(dipole_count):
        dipole = source.dipole_ensemble[n]
        # trace rays then use detector
        for i in range(len(dipole.ray_list)):
            ray = dipole.ray_list[i]
            trace_single_ray(elements, ray)
        detector.add_field(dipole)

    pupil_plot = graphics.PupilPlotDetector(detector)
    return pupil_plot

def trace_rays_mp(system, source, processes=4, binning_detector=False):
    elements = system['elements']
    # decide whether to use ray radius (height) or theta angle
    print("Tracing with %i CPU processes" % processes)
    max_radius=None
    if count_lenses(elements) % 2 == 0:
        is_curved_pupil = True
        max_radius=source.NA
        print("Curved pupil")
    else:
        is_curved_pupil = False
        print("Flat pupil")

    elements = get_first_and_last_lens(elements)

    dipole_count = len(source.dipole_ensemble)

    get_pupils_mp_notf = partial(trace_dipole_rays_mp_notf, elements)

    # plot phis and radii
    dp = source.dipole_ensemble[0]
    dp_raylist = dp.ray_list
    phi_in = [None]*len(dp_raylist)
    r_in = [None]*len(dp_raylist)
    for n in range(len(dp_raylist)):
        ray_ = dp_raylist[n]
        phi_in[n] = ray_.phi
        r_in[n] = np.sin(ray_.theta)

    # split tasks according to processes:
    parallel_start = time.time()
    with Pool(processes) as p:
        source.dipole_ensemble = p.map(get_pupils_mp_notf, source.dipole_ensemble)

    # now sum the fields (single process)
    parallel_time = time.time() - parallel_start
    print("Time elapsed in parallel section (time.time()) %fs" % parallel_time)

    add_start = time.process_time()

    if binning_detector:
        detector = bin_det.BinningDetector(curved=is_curved_pupil)
        # TODO pass dipole source directly and have detector handle this part
        # for n in range(dipole_count):
        #     detector.add_field(source.dipole_ensemble[n])
        detector.add_fields(source)
        detector.bin_intensity()
        detector.integrate_pupil() # compute pupil energy (integrate)
        # detector.interpolate_intensity()
    else:
        detector = det.Detector(curved=is_curved_pupil)
        for n in range(dipole_count):
            detector._add_field_fast(source.dipole_ensemble[n])
        detector.interpolate_intensity()

    add_time = time.process_time() - add_start
    print("Time elapsed in addition %fs" % add_time)
    pupil_plot = graphics.PupilPlotDetector(detector)

    output = SystemOutput(detector, pupil_plot)

    return output