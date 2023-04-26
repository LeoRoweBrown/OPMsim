import optical_matrices
import numpy as np
import shelve

def normalize(v, axis=1):
    print(v.shape)
    norm = np.linalg.norm(v, axis=axis).reshape(v.shape[0],1,1)
    norm[norm == 0] = 1
    return v/norm

def rand_k():
    return 2*np.random.random() - 1

print("Test reflection code")

filename='output/reflection_shelve.out'
my_shelf = shelve.open(filename,'n') # 'n' for new

rand_field = False
rand_kvec = False

k1 = np.array([0.1,-0.5,1])
k2 = np.array([0,-1,1])#
k3 = np.array([-1,0.5,0.5])#

if rand_kvec:
    k1 = np.array([rand_k(), rand_k(), np.random.random()])
    k2 = np.array([rand_k(), rand_k(), np.random.random()])
    k3 = np.array([rand_k(), rand_k(), np.random.random()])


k1 = k1/np.linalg.norm(k1)
k2 = k2/np.linalg.norm(k2)
k3 = k3/np.linalg.norm(k3)


k_vec_norm = np.vstack([k1,k2])#,k1,k2])
k_vec_norm = np.vstack([k1,k2,k3])


E1 = np.array([-1,0,1])
E2 = np.array([0,1,1])
E3 = np.array([-1,0,1])

E_vec = np.vstack([E1,E2])#,E1,E2])
E_vec = np.vstack([E1,E2,E3])

E_vec = E_vec.reshape(E_vec.shape[0], 3, 1)
k_vec_norm = k_vec_norm.reshape(k_vec_norm.shape[0], 3, 1)

# check
print("dot product E and k", np.sum(k_vec_norm * E_vec, axis=1))

N = np.array([0,0,-1]).reshape(1,3,1)

print("k_vec_norm", k_vec_norm)
# print(N)

p = np.cross(k_vec_norm, N, axis=1)  # get p vector (kxN)
p = normalize(p)
r = np.cross(k_vec_norm, p, axis=1)  # get r vector (kxp)
r = normalize(r)
print("p", p)
print("r", r)
print("p shape", r.shape)
print("r shape", r.shape)
print("r shape", r[0].shape)
print("r norm", np.linalg.norm(r))
print("p norm", np.linalg.norm(p))


# basis vectors
x = np.array([1,0,0]).reshape(1,3,1)
y = np.array([0,1,0]).reshape(1,3,1)
z = np.array([0,0,1]).reshape(1,3,1)

# first rotation matrix:
m1 = np.cross(z, k_vec_norm, axis=1)
sin_m1 = np.linalg.norm(m1, axis=1)
theta_m1 = np.arcsin(sin_m1)

m1_unit = normalize(m1, axis=1)
print("m1", m1)

m1_x = m1_unit[:,0]
m1_y = m1_unit[:,1]
m1_z = m1_unit[:,2]

M1 = optical_matrices.arbitrary_rotation(theta_m1, m1_x,m1_y,m1_z)
M1 = M1.reshape(M1.shape[0],3,3)
M1_inv = np.linalg.inv(M1)

E_vec_prime = M1 @ E_vec

Eprime_rand = E_vec_prime
# randomize the field:
for n in range(E_vec.shape[0]):
    Eprime_rand[n, :, :] = np.array([rand_k(), rand_k(), 0]).reshape(3,1)

Eprime =  np.array([
    [1, 0, 0],
    [0,-1,0],
    [-1,-1,0]
    ]).reshape(3,3,1)

print(M1_inv.shape)
print(Eprime_rand.shape)
E_rand = M1_inv @ Eprime_rand
print(E_rand.shape)


if rand_field:
    E_vec = E_rand # np.vstack([E1,E2,E_rand])
else:
    E_vec = M1_inv @ Eprime

# check
print("dot product E and k with rand", np.sum(k_vec_norm * E_vec, axis=1))

r_prime = M1 @ r
r_inv_prime = M1_inv @ r
p_prime = M1 @ p
x_prime = M1 @ x
x_inv_prime = M1_inv @ x

Ep = np.tile([1,0,0], (k_vec_norm.shape[0], 1))
Es = np.tile([0,1,0], (k_vec_norm.shape[0], 1))
kps = np.tile([0,0,1], (k_vec_norm.shape[0], 1))
Ep_ps = np.expand_dims(Ep, axis=[0,3])
Es_ps = np.expand_dims(Es, axis=[0,3])
kps = np.expand_dims(kps, axis=[0,3])

Ep_ps_kz_xyz_ = (np.linalg.inv(M1) @ Ep_ps)
Ep_ps_kz_xyz = Ep_ps_kz_xyz_.squeeze()
Es_ps_kz_xyz = (np.linalg.inv(M1) @ Es_ps).squeeze()

print("r_prime", r_prime)
print("x", x)
m2 = np.cross(r_prime, x, axis=1)
# m2 = np.cross(r, x_inv_prime, axis=1)
# m2 = np.cross(Ep, x)

Ep_ps_kz_xyz_ = Ep_ps_kz_xyz.reshape(r.shape[0],3,1)
print("Ep shape", Ep_ps_kz_xyz_.shape)
print("r shape", r.shape)
m2_mag = np.cross(r, Ep_ps_kz_xyz_, axis=1)
# m2 = M1 @ m2
anglebet_dot = np.sum(r*Ep_ps_kz_xyz_, axis=1)
#anglebet_dot = np.dot(r, Ep_ps_kz_xyz_)
print("r", r)
print("Ep_ps_kz_xyz_", Ep_ps_kz_xyz_)
print("dot", anglebet_dot)
anglebet = np.arccos((anglebet_dot/(np.linalg.norm(r)*np.linalg.norm(Ep_ps_kz_xyz_))))

print("angle between", anglebet)

sin_m2 = np.linalg.norm(m2, axis=1)
theta_m2 = np.arcsin(sin_m2)
m2_unit = normalize(m2)#/np.linalg.norm(m2, axis=1).reshape(m2.shape[0],1,1)
print("theta_m2", theta_m2)

print("m2", m2)
print("sin_m2", sin_m2)

m2_x = m2_unit[:,0]
m2_y = m2_unit[:,1]
m2_z = m2_unit[:,2]

# rotation matrix 2
M2 = optical_matrices.arbitrary_rotation(theta_m2, m2_x,m2_y,m2_z)
M2 = M2.reshape(M2.shape[0],3,3)

M2M1 = M2 @ M1

M2_inv = np.linalg.inv(M2)
M2M1_inv = np.linalg.inv(M2M1)

E_ps = M2M1 @ E_vec
E_zk = M1 @ E_vec

k_ps = M2M1 @ k_vec_norm
k_zk = M1 @ k_vec_norm

print("E_ps", E_ps)
print("E_zk", E_zk)

print("k_ps", k_ps)
print("k_zk", k_zk)

Ep_ps_xyz = (np.linalg.inv(M2M1) @ Ep_ps).squeeze()
Es_ps_xyz = (np.linalg.inv(M2M1) @ Es_ps).squeeze()

kps_xyz = (np.linalg.inv(M2M1) @ kps).squeeze()
kps_kz_xyz = (np.linalg.inv(M1) @ kps).squeeze()

k_kz = (np.linalg.inv(M2) @ kps).squeeze()

print(Es_ps_xyz)
print(Ep_ps_xyz)

print(Es_ps_kz_xyz.shape)

print("M2", M2)
print("M2_inv", M2_inv)

print("kps_kz_xyz", kps_kz_xyz)
print("kps_xyz", kps_xyz)
print("k_vec_norm", k_vec_norm)

from matplotlib import pyplot as plt

Ep_r_dot = np.sum(Ep_ps_xyz.squeeze() * r.squeeze(), axis=1)/np.linalg.norm(Ep_ps_xyz, axis=1)
print("Ep_r_dot (should be 1)", Ep_r_dot)

Es_r_dot = np.sum(Es_ps_xyz.squeeze() * r.squeeze(), axis=1)/np.linalg.norm(Es_ps_xyz, axis=1)
print("Es_r_dot (should be 0)", Es_r_dot)

print("k_z", k_zk)
print("k_vec_norm", k_vec_norm)

n_rays = (k_vec_norm.shape[0])

fig = plt.figure()

for i in range(n_rays):
    ax = fig.add_subplot(1,n_rays,i+1, projection='3d')
    ax.quiver(0,0,0, Es_ps_xyz[i,0], Es_ps_xyz[i,1],Es_ps_xyz[i,2], length=0.1)
    ax.quiver(0,0,0, Ep_ps_xyz[i,0], Ep_ps_xyz[i,1],Ep_ps_xyz[i,2], length=0.1)
    ax.quiver(0,0,0, k_vec_norm[i,0], k_vec_norm[i,1],k_vec_norm[i,2], color='green', length=0.1)
    ax.quiver(0,0,0, kps_xyz[i,0], kps_xyz[i,1], kps_xyz[i,2], color='purple', length=0.1, linestyle='dashed')
    ax.quiver(0,0,0, kps_kz_xyz[i,0], kps_kz_xyz[i,1], kps_kz_xyz[i,2], color='yellow', length=0.1, linestyle='dashed')
    ax.quiver(0,0,0, k_zk[i,0], k_zk[i,1], k_zk[i,2], color='orange', length=0.1, linestyle='dashed')

    ax.quiver(0,0,0, p[i,0], p[i,1], p[i,2], length=0.1, color='black')
    ax.quiver(0,0,0, r[i,0], r[i,1], r[i,2], length=0.1, color='black')
    ax.quiver(0,0,0, N[0,0], N[0,1],N[0,2], color='red', length=0.1)
    ax.quiver(0,0,0, Ep_ps_kz_xyz[i,0], Ep_ps_kz_xyz[i,1], Ep_ps_kz_xyz[i,2], length=0.1, color='gray', linestyle='dashed')
    ax.quiver(0,0,0, Es_ps_kz_xyz[i,0], Es_ps_kz_xyz[i,1], Es_ps_kz_xyz[i,2], length=0.1, color='gray', linestyle='dashed')

    ax.set_aspect('equal')

"""
idx = 2

ax2 = fig.add_subplot(122, projection='3d')
ax2.quiver(0,0,0, Es_ps_xyz[idx,0], Es_ps_xyz[idx,1],Es_ps_xyz[idx,2], length=0.1)
ax2.quiver(0,0,0, Ep_ps_xyz[idx,0], Ep_ps_xyz[idx,1],Ep_ps_xyz[idx,2], length=0.1)
ax2.quiver(0,0,0, k_vec_norm[idx,0], k_vec_norm[idx,1],k_vec_norm[idx,2], color='green', length=0.1)
ax2.quiver(0,0,0, kps_xyz[idx,0], kps_xyz[idx,1], kps_xyz[idx,2], color='purple', length=0.1, linestyle='dashed')
ax2.quiver(0,0,0, kps_kz_xyz[idx,0], kps_kz_xyz[idx,1], kps_kz_xyz[idx,2], color='yellow', length=0.1, linestyle='dashed')
ax2.quiver(0,0,0, k_zk[idx,0], k_zk[idx,1], k_zk[idx,2], color='orange', length=0.1, linestyle='dashed')


ax2.quiver(0,0,0, p[idx,0], p[idx,1], p[idx,2], length=0.1, color='black')
ax2.quiver(0,0,0, r[idx,0], r[idx,1], r[idx,2], length=0.1, color='black')
ax2.quiver(0,0,0, N[0,0], N[0,1],N[0,2], color='red', length=0.1)
ax2.quiver(0,0,0, Ep_ps_kz_xyz[idx,0], Ep_ps_kz_xyz[idx,1], Ep_ps_kz_xyz[idx,2], length=0.1, color='gray', linestyle='dashed')
ax2.quiver(0,0,0, Es_ps_kz_xyz[idx,0], Es_ps_kz_xyz[idx,1], Es_ps_kz_xyz[idx,2], length=0.1, color='gray', linestyle='dashed')
ax2.set_aspect('equal')
"""
plt.show()

for key in dir():
    try:
        my_shelf[key] = globals()[key]
    except TypeError:
        #
        # __builtins__, my_shelf, and imported modules can not be shelved.
        #
        print('ERROR shelving: {0}'.format(key))
my_shelf.close()

def add_plot():
    ax2.quiver(0,0,0, Es_ps_xyz[idx,0], Es_ps_xyz[idx,1],Es_ps_xyz[idx,2], length=0.1)
    ax2.quiver(0,0,0, Ep_ps_xyz[idx,0], Ep_ps_xyz[idx,1],Ep_ps_xyz[idx,2], length=0.1)
    ax2.quiver(0,0,0, k_vec_norm[idx,0], k_vec_norm[idx,1],k_vec_norm[idx,2], color='green', length=0.1)
    ax2.quiver(0,0,0, kps_xyz[idx,0], kps_xyz[idx,1], kps_xyz[idx,2], color='purple', length=0.1, linestyle='dashed')
    ax2.quiver(0,0,0, kps_kz_xyz[idx,0], kps_kz_xyz[idx,1], kps_kz_xyz[idx,2], color='yellow', length=0.1, linestyle='dashed')
    ax2.quiver(0,0,0, k_zk[idx,0], k_zk[idx,1], k_zk[idx,2], color='orange', length=0.1, linestyle='dashed')


    ax2.quiver(0,0,0, p[idx,0], p[idx,1], p[idx,2], length=0.1, color='black')
    ax2.quiver(0,0,0, r[idx,0], r[idx,1], r[idx,2], length=0.1, color='black')
    ax2.quiver(0,0,0, N[0,0], N[0,1],N[0,2], color='red', length=0.1)
    ax2.quiver(0,0,0, Ep_ps_kz_xyz[idx,0], Ep_ps_kz_xyz[idx,1], Ep_ps_kz_xyz[idx,2], length=0.1, color='gray', linestyle='dashed')
    ax2.quiver(0,0,0, Es_ps_kz_xyz[idx,0], Es_ps_kz_xyz[idx,1], Es_ps_kz_xyz[idx,2], length=0.1, color='gray', linestyle='dashed')
    ax2.set_aspect('equal')
