from time import time
from warnings import warn
import numpy as np
from matplotlib import pyplot as plt
from . import matrices
import math

MIN_FIBONACCI_SAMPLES_SPHERE = 10

def uniform_mc_sampler(pdf, input_range, N, maxiter=10000, plot=True):
    """Use Monte Carlo simulation to generate a sample from a PDF"""
    # takes normalised X~pdf(x) function and gets N points according to
    # range of x - {input range[0], input range[1]}
    print("Sampling %d points from PDF (Monte Carlo Rejection method)" % N)
    accepted_points = [0] * N
    input_range[0]
    pdf_x_points = np.linspace(input_range[0], input_range[1], 20)
    plot_norm = np.trapz(pdf(pdf_x_points), pdf_x_points)
    # normalise pdf to 0-1 for p_rand
    norm = np.max(pdf(pdf_x_points))

    n = 0
    i = 0
    while n < N and i < N * maxiter:
        x_rand = input_range[0] + np.random.random() * (input_range[1] - input_range[0])
        p_rand = np.random.random()
        if p_rand < pdf(x_rand) / norm:  # if under curve accept point
            accepted_points[n] = x_rand
            n += 1
        i += 1
    if plot:
        plt.hist(accepted_points, bins=pdf_x_points, density=True)
        plot_x = np.linspace(input_range[0], input_range[1], 20)
        plt.plot(pdf_x_points, pdf(plot_x) / plot_norm, label="PDF")
        plt.legend()
        plt.xlabel(r"$\theta$ (rad)")
        plt.ylabel("Normalised frequency")
    return np.array(accepted_points)


def uniform_points_on_sphere(max_half_angle=(np.pi / 2),
                             point_count=5000,
                             method="rings_phi_inbetween",
                             hemisphere=True):
    """
    Get equal area elements in rings for uniform rays, also compute their area.
    Legacy method that I used to check anisotropy is correct and is symmetric.
    Looks like Fibonacci method has radially asymmetric residuals between theory and sim anisotropy

    """

    if hemisphere:
        N = point_count * 2
    else:
        print("Full sphere generation")
        N = point_count

    region_area = 4 * np.pi / N
    theta_c = np.arccos(1 - 2 / N)
    delta_ideal = (4 * np.pi / N) ** 0.5
    n_collars_ideal = (np.pi - 2 * theta_c) / delta_ideal
    n_collars_fitting = int(np.max([1, np.round(n_collars_ideal)]))
    delta_fitting = delta_ideal * n_collars_ideal / n_collars_fitting

    # areas labelled j=1 to n+2 where n is number of collars,
    # collars are j=2 to n+1, caps are j=1 and n+2
    A_j = [
        2 * np.pi * (np.cos(theta_c + (j - 2) * delta_fitting) - np.cos(theta_c + (j - 1) * delta_fitting))
        for j in range(2, (n_collars_fitting + 1) + 1)
    ]

    area_cap = np.pi * theta_c * theta_c
    total_area = np.sum(A_j) + 2 * area_cap

    n_cells_ideal = np.array(A_j) / region_area

    aj = 0
    n_cells_fitting = np.zeros(n_collars_fitting)
    for j in range(n_collars_fitting):
        n_cells_fitting[j] = np.round(n_cells_ideal[j] + aj)
        aj = np.sum(n_cells_ideal[0: j + 1] - n_cells_fitting[0: j + 1])

    n_cells_fitting = np.concatenate([[1], n_cells_fitting])
    thetas = [np.arccos(1 - (2 / N) * np.sum(n_cells_fitting[0: j + 1])) for j in range(0, n_collars_fitting + 1)]

    n_cells_fitting = np.asarray(n_cells_fitting, dtype=int)

    areas = []
    for i in range(len(A_j)):
        area = A_j[i] / n_cells_fitting[i + 1]
        areas += [area] * n_cells_fitting[i + 1]  # i+1 because cap

    # Scale the surface to match the NA (scale down)
    thetas = np.array(thetas)
    needed_max_theta = max_half_angle
    # get closest match to NA
    max_theta_idx = np.min(np.where(thetas > needed_max_theta))

    theta_scaling = needed_max_theta / thetas[max_theta_idx]

    if hemisphere:
        thetas *= theta_scaling
        thetas = thetas[0: max_theta_idx + 1]
    N_rings = len(thetas) - 1

    area_cap_scaled = np.pi * np.sin(thetas[0]) * np.sin(thetas[0])

    phi_vals_on_ring = []
    phi_k = np.array([0])
    theta_k = np.array([0])
    areas_alt_k = np.array([area_cap_scaled])
    areas_usingcaps = np.array([area_cap_scaled])
    last_phi = [0, 0]  # azimuthal angle

    for i in range(N_rings):  # len - 1, because points are between i and i+1
        dtheta = thetas[i + 1] - thetas[i]
        theta = (thetas[i + 1] + thetas[i]) / 2
        circumference = 2 * np.pi * np.sin(theta)

        # accounts for curvature (correct for all dtheta)
        area_cap_method = 2 * np.pi * (np.cos(thetas[i]) - np.cos(thetas[i + 1])) / n_cells_fitting[i + 1]

        phi_vals = np.linspace(0, 2 * np.pi, n_cells_fitting[i + 1], endpoint=False)

        # determine arrangement of points in each ring - to space as much as possible
        if method == "rings_rotate_gradual":
            phi_vals = (phi_vals + (i / (N_rings + 1)) * np.pi) % (2 * np.pi)
        elif method == "rings_rotate_90":
            phi_vals = (phi_vals + (i % 2) * (np.pi / 2)) % (2 * np.pi)
        elif method == "rings_rotate_random":
            phi_vals = (phi_vals + (np.random.random()) * (np.pi / 2)) % (2 * np.pi)
        elif method == "rings_phi_inbetween":
            if i > 0:
                # offset next ring in phi so that edge of 1st element of the ring is centred
                # on middle of the 1st element of the previous ring. Think like stacking bricks
                offset = (last_phi[0] + last_phi[1]) / 2
                phi_vals = (phi_vals + offset) % (2 * np.pi)
        last_phi = phi_vals  # keep track of last phi so we can place the next point in-between
        phi_vals_on_ring.append(phi_vals)
        phi_k = np.append(phi_k, phi_vals)
        theta_k = np.append(theta_k, [theta] * n_cells_fitting[i + 1])
        areas_usingcaps = np.append(areas_usingcaps, [area_cap_method] * n_cells_fitting[i + 1])

    costheta = (1 - np.sin(max_half_angle)**2) ** 0.5
    expected_area = 2 * np.pi * (1 - costheta)

    return (phi_k, theta_k, areas_usingcaps)

def fibonacci_dipole_generation(point_count=1000):
    """
    Generate uniform spherical dist of dipoles using Fibonacci sphere method. 

    Args:
        point_count (int, optional): Number of points to generate. Defaults to 1000.

    Raises:
        ValueError: If point count is too low for reasonable distribution (MIN_FIBONACCI_SAMPLES_SPHERE)

    Returns:
        tuple: phi, theta, areas: Azimuth, polar angles and the area of each element
    """
    if point_count < MIN_FIBONACCI_SAMPLES_SPHERE:  # reasonable limitation
        raise ValueError(f"At least {MIN_FIBONACCI_SAMPLES_SPHERE} dipoles required for Fibonacci ensemble")
    phi, theta, areas = fibonacci_sphere(max_half_angle=2 * np.pi, point_count=point_count, full_sphere=True)
    return phi, theta, areas

def fibonacci_ray_generation(max_half_angle=(np.pi / 2), point_count=1000):
    """Generate rays using fibonacci method (calls fibonacci_sphere)

    Args:
        max_half_angle (tuple, optional): max half polar angle of sphere pi/2 -> hemisphere. Defaults to (np.pi / 2)
        point_count (int, optional): Number of points to generate. Defaults to 1000.
        show_plot (bool, optional): _description_. Defaults to False.

    Returns:
        tuple: phi, theta, areas: Azimuth, polar angles and the area of each element

    """
    phi, theta, areas = fibonacci_sphere(max_half_angle, point_count * 2, full_sphere=False)
    # mask_theta = np.logical_and(theta > 0, theta < max_half_angle)
    # theta = theta[mask_theta]
    # phi = phi[mask_theta]
    # areas = areas[mask_theta]
    if len(theta) == 0:  # if NA is so small and point count so low that mask masks out all rays
        theta = np.zeros(1)
        phi = np.zeros(1)
        areas = np.ones(1) * 2 * np.pi * (1 - np.cos(max_half_angle))
    return phi, theta, areas

def fibonacci_sphere(max_half_angle,
                     point_count,
                     full_sphere=False,
                     show_plot=False):
    """
    Generate points on a sphere for dipoles and rays.

    Args:
        max_half_angle (tuple, optional): max half polar angle of sphere pi/2 -> hemisphere. Defaults to (np.pi / 2).
        point_count (int, optional): Number of points to generate. Defaults to 1000.
        full_sphere (bool, optional): . Defaults to True.
        show_plot (bool, optional): _description_. Defaults to False.

    Returns:
        tuple: (phi_k, theta_k, areas)
    """
    points = np.zeros((point_count, 3))
    phi = np.pi * (np.sqrt(5.0) - 1.0)  # golden angle in radians

    if point_count < 20:
        warn(f"point_count of {point_count} too low to give reasonably uniform distribution!")
    if point_count == 0:
        return np.array([]), np.array([]), np.array([])
    elif point_count == 1:
        points[0, :] = [1, 0, 0]  # just do x-dipole
    else:
        for i in range(point_count):
            y = 1 - (i / float(point_count - 1)) * 2  # y goes from 1 to -1
            radius = np.sqrt(1 - y * y)  # radius at y

            theta = phi * i  # golden angle increment

            x = np.cos(theta) * radius
            z = np.sin(theta) * radius

            points[i, :] = [x, y, z]

    if not full_sphere:
        mask = np.arccos(points[:, 2]) < max_half_angle
        points = points[mask, :]

    # cartesian to polar
    theta_k = np.arccos(points[:, 2])
    phi_k = np.arctan2(points[:, 1], points[:, 0])

    # calculate area on unit sphere associated with each ray
    # 2*np.pi*(1-costheta) is cap area
    sintheta = np.sin(max_half_angle)
    costheta = (1 - sintheta**2) ** 0.5
    areas = np.ones(len(phi_k)) * 2 * np.pi * (1 - costheta) / len(phi_k)

    if show_plot:
        plt.figure()
        plt.scatter(phi_k, theta_k)
        plt.show()

    return phi_k, theta_k, areas
