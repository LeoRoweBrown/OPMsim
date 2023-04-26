import numpy as np
import scipy
import numpy as np

def protected_mirror_fresnel_matrix(theta_i, n_film_data, film_thickness, n_metal_data, wavelength):
    n_film_wl = n_film_data[:,0]/1e9
    n_film = n_film_data[:,1]
    k_film = n_film_data[:,2]

    n_metal_wl = n_metal_data[:,0]/1e9
    n_metal = n_metal_data[:,1]
    k_metal = n_metal_data[:,2]

    n_film_interp = scipy.interpolate.interp1d(n_film_wl, n_film)
    k_film_interp = scipy.interpolate.interp1d(n_film_wl, k_film)

    n_metal_interp = scipy.interpolate.interp1d(n_metal_wl, n_metal)
    k_metal_interp = scipy.interpolate.interp1d(n_metal_wl, k_metal)

    if hasattr(wavelength, "__len__"):
        if len(wavelength == 1):
            wavelength = np.ones(theta_i)*wavelength

    n_film_complex = n_film_interp(wavelength) - 1j*k_film_interp(wavelength)
    n_metal_complex = n_metal_interp(wavelength) - 1j*k_metal_interp(wavelength)

    print("n_metal_complex", n_metal_complex)
    print("n_film_complex", n_film_complex)

    r_s, r_p = compute_fresnel_protected_mirror(theta_i, n_film_complex, film_thickness, n_metal_complex, wavelength)

    print("r_s", r_s)
    print("r_p", r_p)

    print("Rp", np.real(r_p*np.conj(r_p)))
    print("Rs", np.real(r_s*np.conj(r_s)))

    return r_s, r_p
    
def compute_fresnel_protected_mirror(theta_1, n_film, d, n_metal, wavelength):
    if hasattr(n_film, "__len__"):
        n_points = len(n_film)
    else:
        n_points = 1
    n1 = np.ones(n_points)
    n2 = n_film
    n3 = n_metal

    print("theta_1", theta_1)
    print("n2", n2)
    print("n3", n3)
    print("wavelength", wavelength)

    theta_t2 = np.arcsin((n1/n2)*np.sin(theta_1))

    r_1s = (n1*np.cos(theta_1) - n2*np.cos(theta_t2))/\
        (n1*np.cos(theta_1) + n2*np.cos(theta_t2))

    r_1p = (n2*np.cos(theta_1) - n1*np.cos(theta_t2))/\
        (n2*np.cos(theta_1) + n1*np.cos(theta_t2));

    t_1s = 2*n1*np.cos(theta_1)/(n1*np.cos(theta_1) + n2*np.cos(theta_t2));
    t_1p = 2*n1*np.cos(theta_1)/(n2*np.cos(theta_1) + n1*np.cos(theta_t2));

    # n3 is complex
    theta_t3 = np.arcsin((n2/n3)*np.sin(theta_t2));
    theta_r3 = theta_t2; 

    r_2s = (n2*np.cos(theta_t2) - n3*np.cos(theta_t3))/\
        (n2*np.cos(theta_t2) + n3*np.cos(theta_t3));
    r_2p = (n3*np.cos(theta_t2) - n2*np.cos(theta_t3))/\
        (n3*np.cos(theta_t2) + n2*np.cos(theta_t3));

    phi = 2*np.pi/wavelength * 2 *d*np.real(n2)*np.cos(theta_t2);
    phi_2s = np.angle(r_2s);
    phi_2p = np.angle(r_2p);

    t_3s = 2*n2*np.cos(theta_t2)/(n2*np.cos(theta_t2) + n3*np.cos(theta_t3));
    t_3p = 2*n2*np.cos(theta_t2)/(n3*np.cos(theta_t2) + n2*np.cos(theta_t3));

    r_12s = (n2*np.cos(theta_t2) - n1*np.cos(theta_1))/\
        (n2*np.cos(theta_t2) + n1*np.cos(theta_1))
    r_12p = (n1*np.cos(theta_t2) - n2*np.cos(theta_1))/\
        (n1*np.cos(theta_t2) + n2*np.cos(theta_1))

    # now going from sio2 to air

    t_21s = 2*n2*np.cos(theta_t2)/(n2*np.cos(theta_t2) + n1*np.cos(theta_1));
    t_21p = 2*n2*np.cos(theta_t2)/(n1*np.cos(theta_t2) + n2*np.cos(theta_1));

    # rs_total = r_1s + np.abs(r_2s)*np.exp(1j*(phi_2s + phi))/(1 + r_1p*abs(r_2s)*np.exp(1j*(phi_2s + phi)));
    # rp_total = r_1p + np.abs(r_2p)*np.exp(1j*(phi_2p + phi))/(1 + r_1p*abs(r_2p)*np.exp(1j*(phi_2p + phi)));

    # chris method:
    beta = 2*np.pi*d/wavelength*np.real(n2)*np.cos(theta_t2);

    rp_total = r_1p + t_1p*t_21p*r_2p*np.exp(-1j*2*beta)/(1-r_2p*r_12p*np.exp(-1j*2*beta));
    rs_total = r_1s + t_1s*t_21s*r_2s*np.exp(-1j*2*beta)/(1-r_2s*r_12s*np.exp(-1j*2*beta));

    return rs_total, rp_total

n_film_data = np.genfromtxt('refractive_index_data/SiO2.txt', delimiter='\t')
n_film_data = n_film_data[1:,:]  # remove headers
film_thickness = 158e-9
n_metal_data = np.genfromtxt('refractive_index_data/Ag.txt', delimiter='\t')
n_metal_data = n_metal_data[1:,:]

theta_i = np.linspace(0, np.pi/2, 100)#np.arange(0,np.pi/3, np.pi/20)
wavelength= 500e-9

r_s, r_p = protected_mirror_fresnel_matrix(theta_i, n_film_data, film_thickness, n_metal_data, wavelength)

theta_i = np.linspace(0, np.pi/2, 100)
n_air = 1
n_glass = 1.5

#r_s, r_p = compute_fresnel_protected_mirror(theta_i, n_glass, 1e-3, n_air, wavelength)
Rp = np.real(r_p*np.conj(r_p))
Rs = np.real(r_s*np.conj(r_s))
#print("Rp", np.real(r_p*np.conj(r_p)))
#print("Rs", np.real(r_s*np.conj(r_s)))

from matplotlib import pyplot as plt
plt.figure()
plt.plot(Rp)
plt.plot(Rs)
plt.show()