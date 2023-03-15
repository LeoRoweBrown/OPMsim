# Prototyping mirror transformation code

import numpy as np
import matplotlib.pyplot as plt

def rot_axis_matrix(u_in, sin_m1, cos_m1, inv=False):
    print("sin", sin_m1, "cos", cos_m1, "u_in", u_in)
    mat = np.array([
    [cos_m1+u_in[0]**2*(1-cos_m1), u_in[0]*u_in[1]*(1-cos_m1)-u_in[2]*sin_m1, u_in[0]*u_in[2]*(1-cos_m1)+u_in[1]*sin_m1],
    [u_in[1]*u_in[0]*(1-cos_m1)+u_in[2]*sin_m1, cos_m1 + u_in[1]**2*(1-cos_m1), u_in[1]*u_in[2]*(1-cos_m1)-u_in[0]*sin_m1],
    [u_in[2]*u_in[0]*(1-cos_m1)-u_in[1]*sin_m1, u_in[2]*u_in[1]*(1-cos_m1)+u_in[0]*sin_m1, cos_m1+u_in[2]**2*(1-cos_m1)]
    ])

    if inv:
        mat = np.linalg.inv(mat)
    # print(mat)
    # print("determinant", np.linalg.det(mat))

    return mat

def fresnel_matrix(theta_i, n1=1.5, n2=1):
    # default is glass to air
    # n1sin1 = n2sin2
    sin_theta_t = (n1/n2)*np.sin(theta_i)
    print("sin_theta_t", sin_theta_t)
    print("n2/n1", n2/n1)
    if n1>n2 and sin_theta_t >= 1:
        print("Total internal reflection!")
        theta_t = np.pi/2
    else:
        theta_t = np.arcsin(sin_theta_t)

    print("theta_i", theta_i, theta_i*180/np.pi, "theta_t", theta_t, theta_t*180/np.pi)
    r_s = (n1*np.cos(theta_i) - n2*np.cos(theta_t))/\
        (n1*np.cos(theta_i) + n2*np.cos(theta_t))
    r_p = (n2*np.cos(theta_i) - n1*np.cos(theta_t))/\
        (n2*np.cos(theta_i) + n1*np.cos(theta_t))
    mat = np.array([
        [r_p, 0, 0],
        [0, r_s, 0],
        [0, 0, 1]
    ])
    return mat

def basic_fresnel(theta_i, n1=1.5, n2=1):
    sin_theta_t = (n1/n2)*np.sin(theta_i)
    if n1>n2 and sin_theta_t >= 1:
        theta_t = np.pi/2
    else:
        theta_t = np.arcsin(sin_theta_t)
    r_s = (n1*np.cos(theta_i) - n2*np.cos(theta_t))/\
        (n1*np.cos(theta_i) + n2*np.cos(theta_t))
    r_p = (n2*np.cos(theta_i) - n1*np.cos(theta_t))/\
        (n2*np.cos(theta_i) + n1*np.cos(theta_t))
    return r_p, r_s
    

def reflect_ray_titled_mirror(u_in=None, N=None, E_in=None, n1=None, n2=None, plot=False):
    if N is None:
        normal_angle = 20*np.pi/180
        Nz = -1
        Nx = (Nz)*np.tan(normal_angle)
        N = np.array([Nx,0,Nz])
        print("normal angle (rotation about y axis)", normal_angle)
    print()
    print("------------------------------------")
    print()

    x_unit = np.array([1,0,0])
    y_unit = np.array([0,1,0])
    z_unit = np.array([0,0,1])

    # ray vector
    if u_in is None:
        u_in = np.array([-1,0,1])
    u_in = u_in/np.linalg.norm(u_in)

    # E vector, Ex = Ez
    if E_in is None:
        E_in = np.array([1,1,1])
    # E_in = np.array([0,1,0])

    # check
    print("dot of E and u", np.dot(E_in, u_in))

    # normal vector, nominally (0,0,-1)
    N = N/np.linalg.norm(N)
    
    print("Normal N", N)

    # relfection plane normal and s wave in new coordinate space
    p = np.cross(u_in, N)
    print("p vector", p)

    # 3rd basis vector for p wave in new coordinate space
    r = np.cross(u_in, p)
    print("r vector", r)

    # 1st rotation axis
    m1 = np.cross(u_in, z_unit) #np.cross(u_in, z_unit)
    print("m1",m1)
    sin_m1 = np.linalg.norm(m1)
    cos_m1 = np.sqrt(1 - sin_m1**2)
    m1_unit = m1/np.linalg.norm(m1)
    # rotation matrix 1
    M1 = rot_axis_matrix(m1_unit, sin_m1, cos_m1, False)

    # apply 1st rotation matrix
    u_in_prime = np.matmul(M1, u_in)
    r_prime = np.matmul(M1, r)
    E_prime = np.matmul(M1, E_in)

    # 2nd rotation axis (z -> zk?)
    # m2 = np.cross(r, u_in_prime)  # My version
    # m2 = np.cross(r_prime, x_unit)  # Chris version
    m2 = np.cross(r_prime, x_unit)  # Chris version

    sin_m2 = np.linalg.norm(m2)
    cos_m2 = np.sqrt(1 - sin_m2**2)
    m2_unit = m2/np.linalg.norm(m2)

    print("u-prime", u_in_prime)
    print("E-prime", E_prime)
    # 2nd rotation matrix (rotate about zk?)
    M2 = rot_axis_matrix(m2_unit, sin_m2, cos_m2, False)

    u_ps = np.matmul(M2, u_in_prime)

    M1M2 = np.matmul(M2,M1)

    E_ps = np.matmul(M1M2, E_in)
    print("E_ps", E_ps)

    print("u_ps", u_ps)

    print("E_ps dot u_ps (should be zero still)", np.dot(E_ps, u_ps))

    ## Now do the reflection rotation ##

    #  u = u - 2(N.u)N
    #  E = E - 2(N.E)N

    # frenel coefficients:

    # Express N in this space
    N_ps = np.matmul(M1M2, N)
    print("N_ps", N_ps)

    # calculate rotation matrix to represent mirror 
    p_ps = np.matmul(M1M2, p)
    print("p_ps", p_ps)

    # ray cross N for rotation
    uxN = np.cross(u_ps, N_ps)
    sin_mr_1theta = np.linalg.norm(uxN)
    theta_i = np.arcsin(sin_mr_1theta)
    rot_theta = np.pi-2*theta_i
    cos_mr = np.cos(rot_theta)
    sin_mr = np.sin(rot_theta)
    uxN_unit = uxN/np.linalg.norm(uxN)
    Mr = rot_axis_matrix(uxN_unit, sin_mr, cos_mr, False)

    if n1 is None or n2 is None:
        Mf = np.identity(3)
    else:
        print("theta_i", theta_i)
        Mf = fresnel_matrix(theta_i, n1, n2)
    
    print(Mf)
    # fresnel coefficients applied
    r_p = Mf[0,0]
    r_s = Mf[1,1]

    E_ps = np.matmul(Mf, E_ps)

    u_ps_r2 = u_ps - 2*(np.dot(N_ps, u_ps))*N_ps
    # E_ps_r = E_ps - 2*(np.dot(N_ps, E_ps))*N_ps

    u_ps_r = np.matmul(Mr, u_ps)
    E_ps_r = np.matmul(Mr, E_ps)

    print("u_ps_r", u_ps_r)
    print("E_ps_r", E_ps_r)

    print("reflected dot", np.dot(u_ps_r, E_ps_r))

    # Inverse transform:
    M1M2_inv = np.linalg.inv(M1M2) 
    E_r = np.matmul(M1M2_inv, E_ps_r)
    u_r = np.matmul(M1M2_inv, u_ps_r)
    u_r2 = np.matmul(M1M2_inv, u_ps_r2)

    ## now reflect the ray without transformations to compare ##

    # u_in_r = u_in - 2*(np.dot(N, u_in))*N
    # E_in_r = E_in - 2*(np.dot(N, E_in))*N

    # ray cross N for rotation
    uxN_in = np.cross(u_in, N)
    sin_mr_1theta_in = np.linalg.norm(uxN_in)
    rot_theta_in = np.pi-2*np.arcsin(sin_mr_1theta_in)
    cos_mr_in = np.cos(rot_theta_in)
    sin_mr_in = np.sin(rot_theta_in)
    uxN_in_unit = uxN_in/np.linalg.norm(uxN_in)
    Mr_in = rot_axis_matrix(uxN_in_unit, sin_mr_in, cos_mr_in, False)

    u_in_r = np.matmul(Mr_in, u_in)
    E_in_r = np.matmul(Mr_in, E_in)

    ## results:

    print("----- E-field -----")
    print("E_in", E_in)
    print("E_in_r", E_in_r)
    print("E_r", E_r)

    print("----- k-field -----")
    print("u_in_r", u_in_r)
    print("u_r", u_r)
    print("u_r2", u_r2)

    # draw rays for visualisation:
    len_ray = 1
    p0 = np.array([0,0,0])
    p1 = p0 + u_in*len_ray
    p2 = p1 + u_in_r*len_ray

    # print("dist1", np.linalg.norm(u_in*len_ray))
    # print("dist2", np.linalg.norm(p2-p1))
    if plot:
        ax = plt.figure().add_subplot(projection='3d')

        x_q = [p0[0], p1[0], p2[0]]
        y_q = [p0[1], p1[1], p2[1]]
        z_q = [p0[2], p1[2], p2[2]]

        x_E = [E_in[0], 0, E_r[0]]
        y_E = [E_in[1], 0, E_r[1]]
        z_E = [E_in[2], 0, E_r[2]]

        # x_E = [E_in[0], 0, E_in_r[0]]
        # y_E = [E_in[1], 0, E_in_r[1]]
        # z_E = [E_in[2], 0, E_in_r[2]]


        ax.quiver(x_q, y_q, z_q, x_E, y_E, z_E, length=0.05, normalize=True)
        ax.quiver(x_q[1], y_q[1], z_q[1], N[0], N[1], N[2], length=0.2, normalize=True)
        ax.plot(x_q, y_q, z_q)
        ax.axis('equal') 

        plt.show()
    return r_p, r_s, theta_i

n1 = 1.5
n2 = 1
normal_angle = 10*np.pi/180
theta_0 = 23.69*np.pi/180
Nz = -1
Nx = -(Nz)*np.tan(normal_angle)
N = np.array([Nx,0,Nz])
uz = 1
ux = uz*np.tan(theta_0)
u_in = np.array([ux, 0, uz])
Ez = 1
Ex = -Ez/np.tan(theta_0)
E_in = np.array([Ex, 0, Ez])
r_p, r_s, theta_i = reflect_ray_titled_mirror(u_in=u_in, N=N, E_in=E_in, n1=n1, n2=n2, plot=True)
print("Expected angle:", theta_0+normal_angle, (theta_0+normal_angle)*180/np.pi)
print("Real angle", theta_i, theta_i*180/np.pi)
r_p_basic, r_s_basic = basic_fresnel(theta_0+normal_angle, n1, n2)
print("r_p_basic", r_p_basic, "r_s_basic", r_s_basic)