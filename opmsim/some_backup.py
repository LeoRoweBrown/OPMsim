import scipy
import numpy as np

function [rp_total, rs_total] = fresnel_coefficients_sio2_coatedmirror(theta_1, d, lambda_i)

    n_film_data = np.genfromtxt('refractive_index_data/SiO2.txt', delimiter='\t')
    n_film_data = n_film_data[1:,:]  # remove headers
    d = 158e-6
    n_metal_data = np.genfromtxt('refractive_index_data/Ag.txt', delimiter='\t')
    n_metal_data = n_metal_data[1:,:]

    n_sio2 = interp1(sio2_file(:, 1), sio2_file(:, 2), lambda_i) - 1i*interp1(sio2_file(:, 1), sio2_file(:, 3), lambda_i)
    n_ag = interp1(ag_file(:, 1), ag_file(:, 2), lambda_i) - 1i*interp1(ag_file(:, 1), ag_file(:, 3), lambda_i)

    n_sio2_interp = scipy.interpolate.interp1d()


    lambda_i = lambda_i'*1e-9'

    n1 = 1

    n2 = n_sio2
    n3 = n_ag

    n1 = ones(numel(n2), 1)

    %%
    theta_1 = ones(numel(n1),1)*theta_1
    theta_t2 = asin((n1/n2)*sin(theta_1))

    r_1s = (n1*cos(theta_1) - n2*cos(theta_t2))/...
        (n1*cos(theta_1) + n2*cos(theta_t2))

    r_1p = (n2*cos(theta_1) - n1*cos(theta_t2))/...
        (n2*cos(theta_1) + n1*cos(theta_t2))

    t_1s = 2*n1*cos(theta_1)/(n1*cos(theta_1) + n2*cos(theta_t2))
    t_1p = 2*n1*cos(theta_1)/(n2*cos(theta_1) + n1*cos(theta_t2))

    % n3 is complex
    theta_t3 = asin((n2/n3)*sin(theta_t2))
    theta_r3 = theta_t2 

    r_2s = (n2*cos(theta_t2) - n3*cos(theta_t3))/...
        (n2*cos(theta_t2) + n3*cos(theta_t3))
    r_2p = (n3*cos(theta_t2) - n2*cos(theta_t3))/...
        (n3*cos(theta_t2) + n2*cos(theta_t3))

    % phi = 2*pi/lambda * 2 *d*real(n2)*cos(theta_t2)
    % phi_2s = angle(r_2s)
    % phi_2p = angle(r_2p)

    % t_3s = 2*n2*cos(theta_t2)/(n2*cos(theta_t2) + n3*cos(theta_t3))
    % t_3p = 2*n2*cos(theta_t2)/(n3*cos(theta_t2) + n2*cos(theta_t3))

    r_12s = (n2*cos(theta_t2) - n1*cos(theta_1))/...
        (n2*cos(theta_t2) + n1*cos(theta_1))
    r_12p = (n1*cos(theta_t2) - n2*cos(theta_1))/...
        (n1*cos(theta_t2) + n2*cos(theta_1))

    % now going from sio2 to air

    t_21s = 2*n2*cos(theta_t2)/(n2*cos(theta_t2) + n1*cos(theta_1))
    t_21p = 2*n2*cos(theta_t2)/(n1*cos(theta_t2) + n2*cos(theta_1))

    % chris method:
    beta = 2*pi*d/lambda_i*real(n2)*cos(theta_t2)

    rp_total = r_1p + t_1p*t_21p*r_2p*exp(-1i*2*beta)/(1-r_2p*r_12p*exp(-1i*2*beta))
    rs_total = r_1s + t_1s*t_21s*r_2s*exp(-1i*2*beta)/(1-r_2s*r_12s*exp(-1i*2*beta))
end

