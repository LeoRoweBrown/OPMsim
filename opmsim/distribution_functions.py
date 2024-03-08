from time import time
import numpy as np
from matplotlib import pyplot as plt
from . import optical_matrices
import math

def mc_sampler_photoselection(N, excitation_polarisation, maxiter=10000):
    
    # may never use this, can only image being useful in proving different phis (representing different
    # light-sheet approaching from different angles or rotating sample) are the same, and that changing
    # polarisation to z polarised dramatically reduces signal

    # we want this to be able to take a generalised polarisation and use Monte Carlo to generate
    # photoselection, was thinking of using p_vector.excitation_vector for the cos^2 value
    phi_exc, alpha_exc = excitation_polarisation
    # phi_d are generated in the monte carlo
    # cos_d_exc = np.cos(alpha_exc)*np.cos(phi_exc)*np.cos(alpha_d)*np.cos(phi_d) +\
    #     np.cos(alpha_exc)*np.sin(phi_exc)*np.cos(alpha_d)*np.sin(phi_d) +\
    #     np.sin(alpha_exc)*np.sin(alpha_d)
    # mc uses cos_d_exc**2*cos(alpha), where cos_d_exc is a function of phi_d,exc, alpha_d,exc

    print("Sampling %d points for photoselection (modified Monte Carlo Rejection method)" % N)
    accepted_phi_d = np.zeros(N)
    accepted_alpha_d = np.zeros(N)
    # normalise pdf
    phi_range = 2*np.pi
    theta_range = np.pi/2

    n = 0
    i = 0
    while(n < N and i < N * maxiter ):
        phi_d = np.random.random()*phi_range
        alpha_d = np.random.random()*theta_range

        cos_d_exc = np.cos(alpha_exc)*np.cos(phi_exc)*np.cos(alpha_d)*np.cos(phi_d) +\
            np.cos(alpha_exc)*np.sin(phi_exc)*np.cos(alpha_d)*np.sin(phi_d) +\
            np.sin(alpha_exc)*np.sin(alpha_d)

        # products of trig, should be normalised already to 1?
        pdf = np.abs(np.cos(alpha_d)*cos_d_exc)  
        if pdf > 1:
            raise ValueError("PDF greater than 1")
        if pdf < 0:
            raise ValueError("PDF less than 0")
        p_rand = np.random.random()
        if p_rand < pdf:  # if under curve accept point
            accepted_phi_d[n] = phi_d
            accepted_alpha_d[n] = alpha_d
            n += 1
        i += 1
    if i >= N * maxiter:
        raise StopIteration("Too many tries to obtain fully sampled distribution")

    return accepted_phi_d, accepted_alpha_d
    # raise NotImplementedError()

def uniform_mc_sampler(pdf, input_range, N, maxiter=10000, plot=True):
    """Use Monte Carlo simulation to generate a sample from a PDF"""
    # takes normalised X~pdf(x) function and gets N points according to
    # range of x - {input range[0], input range[1]}
    print("Sampling %d points from PDF (Monte Carlo Rejection method)" % N)
    accepted_points = [0]*N
    input_range[0]
    pdf_x_points = np.linspace(input_range[0], input_range[1], 20)
    plot_norm = np.trapz(pdf(pdf_x_points), pdf_x_points)
    # normalise pdf to 0-1 for p_rand
    norm = np.max(pdf(pdf_x_points))

    n = 0
    i = 0
    while(n < N and i < N * maxiter ):
        x_rand = input_range[0]\
            + np.random.random()*(input_range[1]-input_range[0])
        p_rand = np.random.random()
        if p_rand < pdf(x_rand)/norm:  # if under curve accept point
            accepted_points[n] = x_rand
            n += 1
        i += 1
    if plot:
        plt.hist(accepted_points, bins=pdf_x_points, density=True)
        plot_x = np.linspace(input_range[0], input_range[1], 20)
        plt.plot(pdf_x_points, pdf(plot_x)/plot_norm,label='PDF')
        plt.legend()
        plt.xlabel(r'$\theta$ (rad)')
        plt.ylabel('Normalised frequency')
    return np.array(accepted_points)

def uniform_points_on_sphere(NA=1, point_count=5000,\
    method='uniform_phi_inbetween', hemisphere=True):
    """ Get equal area elements in rings for uniform rays, also compute their area"""

    if hemisphere:
        N = point_count*2
    else:
        print("Full sphere generation")
        N = point_count

    region_area = 4*np.pi/N
    theta_c = np.arccos(1- 2/N)
    delta_ideal = (4*np.pi/N)**0.5
    n_collars_ideal = (np.pi - 2*theta_c)/delta_ideal
    n_collars_fitting = int(np.max([1, np.round(n_collars_ideal)]))
    delta_fitting = delta_ideal * n_collars_ideal/n_collars_fitting

    # areas labelled j=1 to n+2 where n is number of collars, 
    # collars are j=2 to n+1, caps are j=1 and n+2
    A_j = [2*np.pi*(np.cos(theta_c + (j-2)*delta_fitting) -\
        np.cos(theta_c + (j-1)*delta_fitting))\
        for j in range(2,(n_collars_fitting+1)+1)]

    area_cap = np.pi*theta_c*theta_c
    total_area = np.sum(A_j)+2*area_cap

    n_cells_ideal = np.array(A_j)/region_area

    aj = 0
    n_cells_fitting = np.zeros(n_collars_fitting)
    for j in range(n_collars_fitting):
        n_cells_fitting[j] = np.round(n_cells_ideal[j] + aj)
        aj = np.sum(n_cells_ideal[0:j+1] - n_cells_fitting[0:j+1])

    n_cells_fitting = np.concatenate([[1], n_cells_fitting])
    thetas = \
        [np.arccos(1 - (2/N)*np.sum(n_cells_fitting[0:j+1])) for j in range(0, n_collars_fitting+1)]

    n_cells_fitting = np.asarray(n_cells_fitting, dtype=int)

    areas = []
    for i in range(len(A_j)):
        area = A_j[i]/n_cells_fitting[i+1]
        areas += [area]*n_cells_fitting[i+1]  # i+1 because cap

    # Scale the surface to match the NA (scale down)
    thetas=np.array(thetas)
    needed_max_theta = np.arcsin(NA)
    # get closest match to NA
    max_theta_idx = np.min(np.where(thetas > needed_max_theta))

    theta_scaling = needed_max_theta/thetas[max_theta_idx]
    if hemisphere:
        thetas *= theta_scaling
        thetas = thetas[0:max_theta_idx+1]
    N_rings = len(thetas)-1

    area_cap_scaled = np.pi*np.sin(thetas[0])*np.sin(thetas[0])

    phi_vals_on_ring = [None]*(N_rings)
    phi_k = np.array([0])
    theta_k = np.array([0])
    area_k = np.array([area_cap_scaled])
    areas_alt_k = np.array([area_cap_scaled])
    areas_usingcaps = np.array([area_cap_scaled])

    for i in range(N_rings):  # len - 1, because points are between i and i+1
        dtheta = (thetas[i+1] - thetas[i])
        theta = (thetas[i+1] + thetas[i])/2
        circumference = 2*np.pi*np.sin(theta)
        ring_area_man = circumference*dtheta
        # dtheta becomes flat shape (approximate for small dtheta)
        area_manual = ring_area_man/n_cells_fitting[i+1]
        # accounts for curvature (correct for all dtheta)
        area_cap_method = 2*np.pi*(np.cos(thetas[i])-np.cos(thetas[i+1]))/n_cells_fitting[i+1]

        phi_vals = np.linspace(0, 2*np.pi, n_cells_fitting[i+1], endpoint=False)

        # determine arrangement of points in each ring - to space as much as possible
        if method == 'uniform_rotate_gradual':
            phi_vals = (phi_vals + (i/(N_rings+1))*np.pi) % (2*np.pi)
        elif method == 'uniform_rotate_90':
            phi_vals = (phi_vals + (i%2)*(np.pi/2)) % (2*np.pi)
        elif method == 'uniform_rotate_random':
            phi_vals = (phi_vals + (np.random.random())*(np.pi/2)) % (2*np.pi)
        elif method == 'uniform_phi_inbetween':
            if i > 0:
                offset = (last_phi[0] + last_phi[1])/2
                phi_vals = (phi_vals + offset) % (2*np.pi)
        last_phi = phi_vals  # keep track of last phi so we can place the next point in-between
        phi_vals_on_ring[i] = phi_vals

        phi_k = np.append(phi_k, phi_vals)
        theta_k = np.append(theta_k, [theta]*n_cells_fitting[i+1])
        area_k = np.append(area_k, [area]*n_cells_fitting[i+1])
        areas_alt_k = np.append(areas_alt_k, [area_manual]*n_cells_fitting[i+1])
        areas_usingcaps = np.append(areas_usingcaps, [area_cap_method]*n_cells_fitting[i+1])

    costheta = (1-NA**2)**0.5
    expected_area =  2*np.pi*(1-costheta)

    print("cap method area sum", np.sum(areas_usingcaps))
    print("expected area sum", expected_area)

    return (phi_k, theta_k, areas_usingcaps)

def fibonacci_sphere_rays(NA=1, samples=1000):
    samples *= 2  # account for hemisphere
    points = np.zeros((samples, 3))
    phi = np.pi * (np.sqrt(5.) - 1.)  # golden angle in radians

    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
        radius = np.sqrt(1 - y * y)  # radius at y

        theta = phi * i  # golden angle increment

        x = np.cos(theta) * radius
        z = np.sin(theta) * radius

        points[i,:] = [x,y,z]
        # points.append((x, y, z))
    
    # cartesian to polar
    # print("points ", points)
    mask = points[:,2] > 0
    theta_k = np.arccos(points[mask,2])
    phi_k = np.arctan2(points[mask,1],points[mask,0])
    mask_NA = np.sin(theta_k) < NA
    plt.figure()
    plt.scatter(phi_k[mask_NA], theta_k[mask_NA])
    plt.show()

    return (phi_k[mask_NA], theta_k[mask_NA])