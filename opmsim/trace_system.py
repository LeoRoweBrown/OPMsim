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

def count_lenses(elements):
    count = 0
    for i, element in enumerate(elements):
        if element.type == 'SineLens':
            count += 1
    return count

# @profile
def trace_rays(elements, source, options):
    max_radius=None
    if count_lenses(elements) % 2 == 0:
        is_curved_pupil = True
        max_radius=source.NA
        print("Curved pupil")
    else:
        is_curved_pupil = False
        print("Flat pupil")

    detector = Detector(curved=is_curved_pupil)

    rays = source.rays

    if options['calculate_entrace_pupil']:
        initial_rays = deepcopy(rays)
    # trace the rays through the system and calculate the 
    # transfer matrix so that we can trace E vectors separately
    # but efficiently at the end
    #keep_history = False
    #if options['draw_rays']:
         #keep_history = True

    for i, element in enumerate(elements):
        rays = element.apply_matrix(rays)#, keep_history=keep_history)
        # plot? 3d vector diagram
        if 'vector_plots' in options:
            if type(options['vector_plots']) is True:
                rays.quiver_plot()
            elif i in options['vector_plots']:
                rays.quiver_plot()

    rays.finalize_rays_coordinates()#inverse_meridional=False)
    rays.remove_escaped_rays()

    rays.transfer_matrix = rays.transfer_matrix.reshape((1,rays.n_final,3,3))
    rays.E_vec = rays.transfer_matrix @ rays.E_vec

    source.density = source.density.reshape((rays.E_vec.shape[0],1,1,1))
    I_vec = np.real(rays.E_vec*np.conj(rays.E_vec))*(source.density)# /rays.n  # scale by ray count
    
    rays.I_total = np.sum(I_vec, axis=0)  # sum over dipoles
    rays.alternative_minimum = np.min(np.concatenate((rays.I_total[:,0], rays.I_total[:,1])))
    rays.final_energy = np.sum(rays.I_total)

    kx = np.sin(rays.theta)*np.cos(rays.phi)
    ky = np.sin(rays.theta)*np.sin(rays.phi)
    kz = np.cos(rays.theta)

    # for plotting with limiting NA
    element = elements[-1]
    i=0
    while element.type != 'SineLens':
        element = elements[-1-i]
        i += 1
    detector.limiting_NA = element.NA
    detector.limiting_D = element.D
    if is_curved_pupil:
        detector.max_r = element.NA/element.n
        detector.max_r = element.NA/element.n
    else:
        detector.max_r = element.D/2

    print("Energy ratio (efficiency):", rays.final_energy/np.sum(rays.initial_energy))
    print("Total energy:", rays.final_energy)
    print("Total energy per dipole:", rays.final_energy/source.n_dipoles)
    detector.energy_per_dipole = rays.final_energy/source.n_dipoles
    detector.energy_ratio = rays.final_energy/np.sum(rays.initial_energy)
    detector.n_dipoles = source.n_dipoles
    detector.Iy_Ix_ratio = np.sum(rays.I_total[:,1])/np.sum(rays.I_total[:,0])
    print("Iy Ix ratio =", detector.Iy_Ix_ratio)
    

    if any(np.abs(rays.I_total[:,2] > 1e-9)):  # check non-zero z comp
        dotp = np.sum(rays.I_total * rays.k_vec, axis=1)
        if any(np.abs(dotp) > 1e-9):
            print("Error in dot product too!")
            print(dotp)
            print("max dot prod error", max(dotp))
        #print(rays.I_total)
        print("k_vec", rays.k_vec)
        print("mag of k_vec", np.sum(rays.k_vec*rays.k_vec, axis=1))
        print("E_vec", rays.E_vec)
        print(rays.transfer_matrix)
        warnings.warn("Iz is non zero in ray's frame!")
        #raise ValueError("Iz is non zero in ray's frame!")

    print("maxr",detector.max_r)
    detector.detect_rays(rays)  # calculate the pupil field

    # do same with initial rays to plot clipped pupil in O1 space
    # all this does is plot the initial field but only for rays that
    # are traced through the system
    
    if options['calculate_entrace_pupil']:
        if options['entrace_pupil_flat']:
            initial_detector = Detector(curved=False)
            initial_rays = elements[0].apply_matrix(initial_rays)
        else:
            initial_detector = Detector(curved=True)
        initial_rays.finalize_rays_coordinates()  # does this for k_vec only

        # set lost rays E to zero in entrance pupil (removed)
        # initial_rays.set_zero_escaped_rays(rays.escaped)

        # remove lost rays to reduce compute time on applying coord tf matrix
        lost_rays = initial_rays.remove_escaped_rays(rays.escaped)
        
        initial_rays.E_vec = initial_rays.transfer_matrix @ initial_rays.E_vec
        # initial_rays.combine_rays(lost_rays)  # add the zeroes back on
        I_vec_initial = np.real(initial_rays.E_vec*np.conj(initial_rays.E_vec))*source.density

        initial_rays.I_total = np.sum(I_vec_initial, axis=0)  # sum over dipoles
        initial_rays.alternative_minimum = rays.alternative_minimum
        initial_detector.detect_rays(initial_rays)

        return detector, initial_detector
    return detector