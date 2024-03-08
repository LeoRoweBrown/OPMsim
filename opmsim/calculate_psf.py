import numpy as np
from scipy.integrate import simpson

def calculate_psf(rays, N=100, size=2e-6):
    """
    Calculate PSF with Debye-Wolf integral
    rays: ray object
    N: number of points for which PSF is calculated in each dimension
    """

    xi = np.linspace(-size, size, N)
    x_vec = np.array([np.linspace(-size, size, N),
             np.linspace(-size, size, N),
             np.linspace(-size, size, N)]).T
    
    x_vec_i, x_vec_j, x_vec_k = np.meshgrid(xi,xi,xi)

    theta = np.arccos(rays.k_vec[:,2]).flatten()
    phi = np.arctan2(rays.k_vec[:,1], rays.k_vec[:,0]).flatten()
    kdotx = rays.k_vec * x_vec
    E_vec_psf = np.zeros((N,N,N))
    for i in range(N):
        for j in range(N):     
            for k in range(N):
                kdotx = rays.k_vec * np.array([x_vec_i[i,k,j], x_vec_j[i,k,j], x_vec_k[i,k,j]])

                integrand = -(1j*rays.k_vec/2*np.pi)*rays.E_vec * np.exp(1j*kdotx) * np.sin(theta)
                E_vec_psf[i, j, k] = simpson(simpson(integrand, x=theta), x=phi)

    return E_vec_psf, size