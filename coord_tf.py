import numpy as np
import optical_matrices

def finalize_coord_system(ray, curved_pupil=False):

    if ray.isMeridional:  # put back into non meridional basis
        ray.E_vec = np.matmul(
            optical_matrices.meridional_transform(ray.phi, inverse=True),\
            ray.E_vec)
        ray.isMeridional = False
    # if curved we evaluate the field over a curved surface, which needs curved coord system

    if abs(ray.theta) > 1e-5 or curved_pupil:  # curved pupil, can be specified explicitly
        ray.E_vec = _rotate_field_curved_surface(ray.E_vec, ray.phi, ray.theta)
        ray.update_history()  # before changing from (-r, phi) -> (r, phi+180)
        ray.theta, ray.phi = get_final_polar_coords(ray.theta, ray.phi)
        if ray.theta < 0:
            raise Warning("negative radius in polar plot!")
        ray.polar_radius = np.sin(ray.theta)#attr=True))
    else:  # flat pupil
        E_vec = ray.E_vec
        # ray_polar_radius[n] = np.sin(ray.get_theta())
        ray.update_history()  # before changing from (-r, phi) -> (r, phi+180)
        ray.rho, ray.phi = get_final_polar_coords(ray.rho, ray.phi)
        ray.polar_radius = ray.rho

    return ray

def _rotate_field_curved_surface(E_vec, phi, theta):
    # rotate into meridional by phi and then in theta so k perpendicular to surface
    # E_vec = deepcopy(E_vec_in) # to be safe
    E_vec = _rotate_efield(E_vec, phi, theta)
    Ex = np.absolute(E_vec[0])
    Ey = np.absolute(E_vec[1])
    Ez = np.absolute(E_vec[2])
    if Ez**2 > 1e-3*(Ex**2 + Ey**2 + Ez**2) and Ez > 1e-9:  # the end bit is a bit fudgy
        print("Ez =", E_vec[2], "Ex =", E_vec[0], "E_y =", E_vec[1])
        raise Exception("E_z is not zero in ray's frame!")

    # now convert x and y rotated meridional basis back to lab basis for meaningful
    # polarisation

    # xy field (polarisation) transformation to recover original x-y basis
    E_vec_x = E_vec[0]*np.cos(phi) - E_vec[1]*np.sin(phi)
    E_vec_y = E_vec[0]*np.sin(phi) + E_vec[1]*np.cos(phi)

    E_vec[0] = E_vec_x
    E_vec[1] = E_vec_y

    return E_vec

def _rotate_efield(E_vec, phi, theta_polar):
    """ 
    changes coordinate system according to k vector so that Ez = 0
    both rotates into meridional and then does theta rotation
    """
    E_x_tf = E_vec[0]*np.cos(phi)*np.cos(theta_polar) + \
        E_vec[1]*np.sin(phi)*np.cos(theta_polar)\
        - E_vec[2]*np.sin(theta_polar)
    E_y_tf = -E_vec[0]*np.sin(phi) + E_vec[1]*np.cos(phi)
    # E_z_tf should equal 0
    E_z_tf = E_vec[0]*np.sin(theta_polar)*np.cos(phi)\
        + E_vec[1]*np.sin(theta_polar)*np.sin(phi) + E_vec[2]*np.cos(theta_polar)
    E_rot = [E_x_tf, E_y_tf, E_z_tf]

    return E_rot

def get_final_polar_coords(ray, curved=True):
    """polar plot does not deal with negative radii, r=theta or rho"""
    phi = ray.phi
    if curved:
        r = abs(np.sin(ray.theta))
    else:
        r = ray.rho
    if ray.rho < 0:
        # TEMPORARY COMMENT OUT
        # phi = (ray.phi + np.pi) % (2*np.pi)
        # phi = (phi + np.pi)
        # go from negative radius to phi = phi + pi
        ## we WOULD use modulo but it gets rid of 2pi, which we need because
        ## interpolation doesn't wrap around from 2pi to 0.
        phi = phi + np.pi
        mask = (phi > 2*np.pi)*1
        phi = phi - mask*2*np.pi
        if not curved:  # abs already takes care of this in curved case
            r = -r
    # general moduluo
    mask = (phi > 2*np.pi)*1  # use this instead of modulu seemed to work better?
    phi = phi - mask*2*np.pi
    return r, phi

# def get_final_polar_coords(r, phi):
#     """polar plot does not deal with negative radii, r=theta or rho"""
#     if r < 0:
#         # TEMPORARY COMMENT OUT
#         phi = (phi + np.pi) % (2*np.pi)
#         # phi = (phi + np.pi)
#        # phi = phi + np.pi
#         # mask = (phi > 2*np.pi)*1
#         # phi = phi - mask*2*np.pi
#         r = -r
#     return r, phi
