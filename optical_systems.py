from copy import deepcopy
import detector
import optical_elements
import trace_system
import dipole_source
from matplotlib import pyplot as plt
import numpy as np
import graphics
import anisotropy
import time


def objective_system(lenses, source_dict=None, processes=2, plot=True, title="OPM"):
    """
    O1, O2, O3 are dictionaries with fields NA, f and source is a dictionary
    with dipole_source and source name
    """


    init_start = time.time()
    if source_dict is None:
        source = dipole_source.DipoleSource()
        source.add_dipoles(0,0)
        source_dict['name'] = 'X-dipole'
        source_dict['source'] = source

    source = source_dict['source']

    O1 = lenses[0]
    source.get_rays_uniform_rings(O1['NA']/O1['n'], O1['f'], ray_count=1000)
    dipoles = source.dipole_ensemble
    dp = dipoles[0]
    rays = dp.ray_list
    print("N rays", len(rays))
    thetas = np.zeros_like(rays)
    for r in range(len(rays)):
        # print("ray")
        thetas[r] = rays[r].theta
    print("Starting thetas", thetas)

    elements = []

    for lens in lenses:
        rot = 0
        print(lens)
        if 'rotation' in lens:
            rot = lens['rotation']*np.pi/180
        print("Lens rotation:", rot)
        elements.append(optical_elements.SineLens(lens['NA']/lens['n'], lens['f'], xAxis_rotation=rot))

    system = {'name': title, 'elements': elements}

    dipole = source.dipole_ensemble[0]
    rays = dipole.ray_list

    init_time = time.time() - init_start
    print("initialisation time in OPM system %fs" % init_time)

    trace_start = time.time()
    output = trace_system.trace_rays_mp(system, source, processes, binning_detector=True)
    trace_time = time.time() - trace_start
    print("time in trace_rays_mp %fs" % trace_time)
    pupil = output.pupil_plot

    # pupil.plot(system['name'], save_dir=None, file_name=None,\
    #     show_prints=False, plot_arrow=None)
    pupil.plot(system['name'], save_dir=None, file_name=None,\
        show_prints=False, plot_arrow=None, projection=None, unstructured_data=True)
    """
    rays = output.detector.dipole_source.dipole_ensemble[0].ray_list
    history = rays[0].ray_history
    print(history)
    for i in range(len(history)):
        print(i, history[i].note)

    theta_prelens = np.zeros(len(rays))
    phi_prelens = np.zeros(len(rays))
    xyz = np.zeros([3, len(rays)])
    rhos = np.zeros(len(rays))

    for n in range(len(rays)):
        theta_prelens[n] =  rays[n].ray_history[1].theta
        phi_prelens[n] = rays[n].ray_history[1].phi
        xyz[0, n] = rays[n].ray_history[1].k_vec[0]
        xyz[1, n] = rays[n].ray_history[1].k_vec[1]
        xyz[2, n] = rays[n].ray_history[1].k_vec[2]
        rhos[n] = rays[n].ray_history[1].rho
        # print("escaped", rays[n].escaped)

    f3d = plt.figure(figsize=[15,8])
    ax = f3d.add_subplot(121, projection='3d')
    ax.scatter(xyz[0],xyz[1],xyz[2])
    ax.set_xlim3d([-1,1])
    ax.set_ylim3d([-1,1])
    ax.set_zlim3d([-1,1])

    f = plt.figure()
    ax = f.add_subplot(121) 
    ax.scatter(rhos*np.cos(phi_prelens), rhos*np.sin(phi_prelens), s=1)
    ax1 = f.add_subplot(122)
    ax1.scatter(np.sin(theta_prelens)*np.cos(phi_prelens), np.sin(theta_prelens)*np.sin(phi_prelens), s=1)
    plt.show()
    """
    return output


def anisotropy_measuring_system(collection_NA, collection_f, excitation_polarisation, source=None, two_obj=False, processes=1):
    init_start = time.time()
    if source is None:
        source = dipole_source.DipoleSource()
        source.generate_dipoles(100, method='uniform_phi_inbetween')
        source.get_rays(collection_NA, collection_f)

    print(source)
    
    elements_p = []
    elements_p.append(optical_elements.SineLens(collection_NA, collection_f))
    elements_p.append(optical_elements.LinearPolariser(excitation_polarisation[0]))
    if two_obj:
        # final_lens=True means don't trace rays to a point after this objective in this case
        elements_p.append(optical_elements.SineLens(collection_NA, collection_f))

    system_p = {'name': 'Parallel arm anisotropy', 'elements': elements_p}
    source.get_rays(collection_NA, collection_f)

    init_time = time.time() - init_start
    print("initialisation time in anisotropy_measuring_system %fs" % init_time)

    trace_start = time.time()
    pupil_p = trace_system.trace_rays_mp(system_p, source, collection_NA, processes)
    trace_time = time.time() - trace_start
    print("time in trace_rays_mp %fs" % trace_time)

    pupil_p.plot(system_p['name'], save_dir=None, file_name=None,\
        show_prints=False, plot_arrow=None)
    pupil_p.plot(system_p['name'], save_dir=None, file_name=None,\
        show_prints=False, plot_arrow=None, projection=None)
    # reset rays/retrace rays
    source.get_rays(collection_NA, collection_f)

    elements_s = []
    elements_s.append(optical_elements.SineLens(collection_NA, collection_f))
    elements_s.append(optical_elements.LinearPolariser(excitation_polarisation[0]+np.pi/2))
    if two_obj:
        elements_s.append(optical_elements.SineLens(collection_NA, collection_f))

    system_s = {'name': 'Perpendicular arm anisotropy', 'elements': elements_s}
    pupil_s = trace_system.trace_rays_mp(system_s, source, collection_NA, processes)

    pupil_s.plot(system_s['name'], save_dir=None, file_name=None,\
        show_prints=False, plot_arrow=None)
    pupil_s.plot(system_s['name'], save_dir=None, file_name=None,\
        show_prints=False, plot_arrow=None, projection=None)

    r = anisotropy.calculate_anisotropy(pupil_p, pupil_s)
    print("Anisotropy r =", r)

def compare_curved_and_flat(collection_NA, collection_f, excitation_polarisation, source=None, processes=1):
    init_start = time.time()
    if source is None:
        source = dipole_source.DipoleSource()
        source.generate_dipoles(100, method='uniform_phi_inbetween')
        source.get_rays(collection_NA, collection_f)

    print(source)
    
    elements_flat = []
    elements_flat.append(optical_elements.SineLens(collection_NA, collection_f))
    # elements_flat.append(optical_elements.LinearPolariser(excitation_polarisation[0]))

    system_flat = {'name': 'Parallel arm anisotropy', 'elements': elements_flat}
    source.get_rays(collection_NA, collection_f)

    pupil_flat = trace_system.trace_rays_mp(system_flat, source, collection_NA, processes)

    pupil_flat.plot(system_flat['name'], save_dir=None, file_name=None,\
        show_prints=False, plot_arrow=None)
    pupil_flat.plot(system_flat['name'], save_dir=None, file_name=None,\
        show_prints=False, plot_arrow=None, projection=None)
    # reset rays/retrace rays
    source.get_rays(collection_NA, collection_f)  # reset

    elements_curved = []
    elements_curved.append(optical_elements.SineLens(collection_NA, collection_f))
    # elements_curved.append(optical_elements.LinearPolariser(excitation_polarisation[0]+np.pi/2))
    elements_curved.append(optical_elements.SineLens(collection_NA, collection_f))

    system_curved = {'name': 'Perpendicular arm anisotropy', 'elements': elements_curved}
    pupil_curved = trace_system.trace_rays_mp(system_curved, source, collection_NA, processes)

    pupil_curved.plot(system_curved['name'], save_dir=None, file_name=None,\
        show_prints=False, plot_arrow=None)
    pupil_curved.plot(system_curved['name'], save_dir=None, file_name=None,\
        show_prints=False, plot_arrow=None, projection=None)

    pupil_curved_scaled = deepcopy(pupil_curved)
    pupil_curved_scaled.scale_intensity_from_curved_pupil()
    pupil_curved_scaled.plot(system_curved['name'] + " from curved, scaled", save_dir=None, file_name=None,\
        show_prints=False, plot_arrow=None)
    
    resid = deepcopy(pupil_flat)
    resid.data_x = (resid.data_x - pupil_curved_scaled.data_x)/resid.data_x
    resid.data_y = (resid.data_x - pupil_curved_scaled.data_y)/resid.data_y

    resid.plot(system_curved['name'] + " RESIDUALS", save_dir=None, file_name=None,\
        show_prints=False, plot_arrow=None)

def compare_single_dipole_theory(collection_NA, collection_f,\
    excitation_polarisation, source=None, processes=1, plot=True,\
    rays_distribution=None, show_ray_distribution=False, binning_detector=False,\
    use_raw_for_anisotropy=True):

    # ray_count = (100,25)
    source = dipole_source.DipoleSource()
    source.add_dipoles(excitation_polarisation[0], excitation_polarisation[1])

    elements_p = []
    elements_p.append(optical_elements.SineLens(collection_NA, collection_f,\
        binning_method=binning_detector))  # tell it that we bin so it doesn't do cosine scaling
    elements_p.append(optical_elements.LinearPolariser(0))
    # elements_p.append(optical_elements.SineLens(collection_NA, collection_f,\
    #    binning_method=binning_detector))  # tell it that we bin so it doesn't do cosine scaling

    system_p = {'name': 'Parallel arm anisotropy', 'elements': elements_p}
    if rays_distribution=='uniform':
        source.get_rays_uniform(collection_NA, collection_f, plot_sphere=show_ray_distribution)
    elif rays_distribution=='uniform_rings':
        source.get_rays_uniform_rings(collection_NA, collection_f,\
            plot_sphere=show_ray_distribution)
    else:
        source.get_rays(collection_NA, collection_f)

    output_p = trace_system.trace_rays_mp(system_p, source,\
        processes, binning_detector=binning_detector)

    pupil_p = output_p.pupil_plot
    if plot:
        pupil_p.plot(system_p['name'], save_dir=None, file_name=None,\
            show_prints=False, plot_arrow=None)
        pupil_p.plot(system_p['name'], save_dir=None, file_name=None,\
            show_prints=False, plot_arrow=None, projection=None)

    # reset rays/retrace rays
    if rays_distribution=='uniform':
        source.get_rays_uniform(collection_NA, collection_f)
    elif rays_distribution=='uniform_rings':
        source.get_rays_uniform_rings(collection_NA, collection_f,\
            plot_sphere=show_ray_distribution)
    else:
        source.get_rays(collection_NA, collection_f)
    if show_ray_distribution:
        source.display_pupil_rays()

    elements_s = []
    elements_s.append(optical_elements.SineLens(collection_NA, collection_f,\
        binning_method=binning_detector))
    elements_s.append(optical_elements.LinearPolariser(np.pi/2))
    # elements_s.append(optical_elements.SineLens(collection_NA, collection_f,\
    #         binning_method=binning_detector))


    system_s = {'name': 'Perpendicular arm anisotropy', 'elements': elements_s}
    output_s = trace_system.trace_rays_mp(system_s, source,\
        processes, binning_detector=binning_detector)

    pupil_s = output_s.pupil_plot
    if plot:
        pupil_s.plot(system_s['name'], save_dir=None, file_name=None,\
            show_prints=False, plot_arrow=None)
        pupil_s.plot(system_s['name'], save_dir=None, file_name=None,\
            show_prints=False, plot_arrow=None, projection=None)
    
    # calculate anisotropy based on graphical pupil (not accurate)
    r = anisotropy.calculate_anisotropy(pupil_p, pupil_s)
    print("Graphical Anisotropy r =", r)

    if output_p.detector.I_total_integral is not None\
        and output_s.detector.I_total_integral and use_raw_for_anisotropy:
        print("Calculating intensity from raw data")
        r = anisotropy.calculate_anisotropy_rawdata(output_p.detector, output_s.detector)
        print("r from raw data =", r)
        # area from NA
        costheta = (1-collection_NA**2)**0.5
        expected_area =  2*np.pi*(1-costheta)
        print("Expected area (curved)", expected_area)
        expected_flat_area = np.pi * collection_NA**2
        print("Expected flat area", expected_flat_area)


    ## anisotropy from Axelrod
    r_theory, I_p, I_s = anisotropy.theoretical_anisotropy(collection_NA, excitation_polarisation, return_intensities=True)
    print("Theoretical anisotropy r =", r_theory)
    print("theory intensities", I_p, I_s)

    data_out = dict()
    data_out['r'] = r
    data_out['r_theory'] = r_theory
    data_out['detector_obj_s'] = output_s.detector
    data_out['detector_obj_p'] = output_p.detector

    return data_out

def compare_population_dipole_theory(collection_NA, collection_f, dipole_count=50, source=None, processes=1, plot=True, rays_distribution=None):
    excitation_polarisation = (0,0)
    source = dipole_source.DipoleSource()

    method='uniform_phi_inbetween'

    source.generate_dipoles(dipole_count, method=method)
    source.classical_photoselection(excitation_polarisation)

    elements_p = []
    elements_p.append(optical_elements.SineLens(collection_NA, collection_f))
    elements_p.append(optical_elements.LinearPolariser(0))

    system_p = {'name': 'Parallel arm anisotropy', 'elements': elements_p}
    if rays_distribution=='uniform':
        source.get_rays_uniform(collection_NA, collection_f)
    elif rays_distribution=='uniform_rings':
        source.get_rays_uniform_rings(collection_NA, collection_f,plot_sphere=True)
    else:
        source.get_rays(collection_NA, collection_f)

    pupil_p = trace_system.trace_rays_mp(system_p, source, collection_NA, processes)
    if plot:
        pupil_p.plot(system_p['name'], save_dir=None, file_name=None,\
            show_prints=False, plot_arrow=None)

    # reset rays/retrace rays
    
    if rays_distribution=='uniform':
        source.get_rays_uniform(collection_NA, collection_f)
    elif rays_distribution=='uniform_rings':
        source.get_rays_uniform_rings(collection_NA, collection_f,plot_sphere=True)
    else:
        source.get_rays(collection_NA, collection_f)

    source.display_pupil_rays()

    elements_s = []
    elements_s.append(optical_elements.SineLens(collection_NA, collection_f))
    elements_s.append(optical_elements.LinearPolariser(np.pi/2))

    system_s = {'name': 'Perpendicular arm anisotropy', 'elements': elements_s}
    pupil_s = trace_system.trace_rays_mp(system_s, source, collection_NA, processes)

    if plot:
        pupil_s.plot(system_s['name'], save_dir=None, file_name=None,\
            show_prints=False, plot_arrow=None)

    r = anisotropy.calculate_anisotropy(pupil_p, pupil_s)
    print("Anisotropy r =", r)

    ## anisotropy from Axelrod, only valid for x dipole
    r_theory, I_p, I_s = anisotropy.theoretical_anisotropy_population(collection_NA, return_intensities=True)
    print("Theoretical anisotropy r =", r_theory)
    print("theory intensities", I_p, I_s)

    return (r, r_theory)

def one_objective(collection_NA, collection_f, excitation_polarisation, source=None, processes=1, plot=True):
    source = dipole_source.DipoleSource()
    source.add_dipoles(excitation_polarisation[0], excitation_polarisation[1])

    elements_p = []
    elements_p.append(optical_elements.SineLens(collection_NA, collection_f))

    system_p = {'name': 'Parallel arm anisotropy', 'elements': elements_p}
    source.get_rays(collection_NA, collection_f)

    pupil_p = trace_system.trace_rays_mp(system_p, source, collection_NA, processes)
    if plot:
        pupil_p.plot(system_p['name'], save_dir=None, file_name=None,\
            show_prints=False, plot_arrow=None)

def compare_methods(original_data, new_data, absolute=True):
    if absolute:
        pc_resid = new_data-original_data
    else:
        pc_resid = (new_data-original_data)/original_data
    pc_resid[np.isnan(pc_resid)] = 0
    return pc_resid