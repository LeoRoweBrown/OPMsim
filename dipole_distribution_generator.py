import numpy as np
from matplotlib import pyplot as plt

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
    # takes normalised X~pdf(x) function and gets N points according to
    # range of x - {input range[0], input range[1]}
    print("Sampling %d points from PDF (Monte Carlo Rejection method)" % N)
    accepted_points = [0]*N
    input_range[0]
    # normalise pdf
    pdf_x_points = np.linspace(input_range[0], input_range[1], 20)
    norm = np.trapz(pdf(pdf_x_points), pdf_x_points)

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
        plt.hist(accepted_points, density=True)
        plot_x = np.linspace(input_range[0], input_range[1], 20)
        plt.plot(pdf_x_points, pdf(plot_x)/norm,label='PDF')
        plt.legend()
        plt.xlabel(r'$\theta$ (rad)')
        plt.ylabel('Normalised frequency')
    return accepted_points