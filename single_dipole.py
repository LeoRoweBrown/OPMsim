
from re import X
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from scipy.interpolate import griddata
import dipole as dipole
from plot_single_dipole import plot_single_dipole
from plot_single_dipole import plot_single_dipole_cartesian
from tools import printif  # is the name tools a little dangerous? conflicts etc

def simulate_single_dipole(dipole_phi, dipole_alpha, NA, flat_pupil=False,
          dipole_obj=None, plot_pupil=None,
          save_dir=None, show_prints=False,
          curved_coords=True):

     if dipole_obj is None:
          single_dipole = dipole.Dipole(dipole_phi, dipole_alpha) # phi then theta
     elif isinstance(dipole_obj, dipole.Dipole):
          single_dipole = dipole_obj
     else:
          raise Exception("Invalid input for dipole_obj, should be instance of Dipole from dipole")

     # NA = n*sin(half_angle)
     max_sin_theta = NA  # assume n = 1
     pupil_sin_theta_range = np.linspace(0,max_sin_theta,50)
     pupil_theta_range = np.arcsin(pupil_sin_theta_range)
     pupil_phi_range = np.linspace(0,2*np.pi,100)

     printif('dipole angle: phi = %f theta = %f' % (single_dipole.phi_d*(180/np.pi),\
          single_dipole.alpha_d*(180/np.pi)), show_prints)

     pupil_vals_x = np.zeros([len(pupil_theta_range), len(pupil_phi_range)])
     pupil_vals_y = np.zeros([len(pupil_theta_range), len(pupil_phi_range)])
     pupil_vals_mag = np.zeros([len(pupil_theta_range), len(pupil_phi_range)],\
          dtype=np.complex_)

     phi_list = np.zeros([len(pupil_theta_range) * len(pupil_phi_range)])
     sin_theta_list = np.zeros([len(pupil_theta_range) * len(pupil_phi_range)])

     # evaluate the field across the curved pupil
     i = 0
     for t_i, theta in enumerate(pupil_theta_range):
          for p_i, phi in enumerate(pupil_phi_range):
               e_vec = np.array([])
               e_mag = 1
               k_vec = np.array([])
               if flat_pupil:
                    e_vec, e_mag, k_vec = single_dipole.getEfield_z(theta, phi, 1)  
                    # flat pupil dont transform coords!
                    # include the prefactor because r is different
                    # e_vec *= e_mag
                    e_vec *= ((np.real(e_mag)**2 + np.imag(e_mag)**2)**0.5)
               else:
                    e_vec, e_mag, k_vec = single_dipole.getEfield(theta, phi, 1, 
                         curved_coords=curved_coords)
               # print(e_vec)
               # print('######')
               pupil_vals_x[t_i, p_i] = np.real(e_vec[0])
               pupil_vals_y[t_i, p_i] = np.real(e_vec[1])
               pupil_vals_mag[t_i, p_i] = e_mag

               phi_list[i] = pupil_phi_range[p_i]
               sin_theta_list[i] = pupil_sin_theta_range[t_i]
               i += 1

     #grid of values, interpolated a bit more?

     grid_sin_theta, grid_phi = np.meshgrid(pupil_sin_theta_range, pupil_phi_range)
     points = (sin_theta_list, phi_list)

     printif('##', show_prints)

     vals_efield_x = pupil_vals_x.flatten()
     vals_efield_y = pupil_vals_y.flatten()

     vals_intensity_x = (vals_efield_x*np.conjugate(vals_efield_x)).real
     vals_intensity_y = (vals_efield_y*np.conjugate(vals_efield_y)).real

     data_x = griddata(points, pupil_vals_x.flatten(), (grid_sin_theta, grid_phi),\
          method='cubic',fill_value=0)
     data_y = griddata(points, pupil_vals_y.flatten(), (grid_sin_theta, grid_phi),\
          method='cubic',fill_value=0)

     data_intensity_x = griddata(points, vals_intensity_x, (grid_sin_theta, grid_phi),\
          method='cubic',fill_value=0)
     data_intensity_y = griddata(points, vals_intensity_y, (grid_sin_theta, grid_phi),\
          method='cubic',fill_value=0)

     if plot_pupil: 
          printif("Plotting...", show_prints)
          plot_single_dipole((pupil_phi_range,pupil_sin_theta_range),\
               (data_intensity_x, data_intensity_y),\
               (single_dipole.phi_d, single_dipole.alpha_d), save_dir)

     return points, vals_efield_x, vals_efield_y


### made entire other function for cartesian testing, will remove

def simulate_single_dipole_cartesian(dipole_phi, dipole_alpha, NA, z=1,
     dipole_obj=None, plot_pupil=None,
     save_dir=None, show_prints=False,
     curved_coords=True):
          
     if dipole_obj is None:
          single_dipole = dipole.Dipole(dipole_phi, dipole_alpha) # phi then theta
     elif isinstance(dipole_obj, dipole.Dipole):
          single_dipole = dipole_obj
     else:
          raise Exception("Invalid input for dipole_obj, should be instance of Dipole from dipole")

     # NA = n*sin(half_angle)
     # NA = n*sin((D/2)/z)
     pupil_r = np.arcsin(NA)*z

     y_range = np.linspace(0, pupil_r, 50)
     x_range = np.linspace(0, pupil_r, 50)

     max_sin_theta = NA  # assume n = 1

     # generate based on cartesian:


     pupil_sin_theta_range = np.linspace(0,max_sin_theta,50)
     pupil_theta_range = np.arcsin(pupil_sin_theta_range)
     pupil_phi_range = np.linspace(0,2*np.pi,100)

     printif('dipole angle: phi = %f theta = %f' % (single_dipole.phi_d*(180/np.pi),\
          single_dipole.alpha_d*(180/np.pi)), show_prints)

     # pupil_vals_x = np.zeros([len(pupil_theta_range), len(pupil_phi_range)])
     # pupil_vals_y = np.zeros([len(pupil_theta_range), len(pupil_phi_range)])
     pupil_vals_mag = np.zeros([len(pupil_theta_range), len(pupil_phi_range)],\
          dtype=np.complex_)

     pupil_vals_x = np.zeros([len(x_range), len(y_range)])
     pupil_vals_y = np.zeros([len(x_range), len(y_range)])

     x_list = np.zeros([len(x_range) * len(y_range)])
     y_list = np.zeros([len(x_range) * len(y_range)])

     # evaluate the field across the curved pupil
     i = 0
     for x_i, x in enumerate(x_range):
          for y_i, y in enumerate(y_range):
               e_vec = np.array([])
               e_mag = np.array([])
               k_vec = np.array([])

               e_vec, e_mag, k_vec = single_dipole.getEfield_z_cartesian(x, y, z)  
               # flat pupil dont transform coords!
               # include the prefactor because r is different
               e_vec *= np.real(e_mag)
               # print(e_vec)
               # print('######')
               pupil_vals_x[x_i, y_i] = e_vec[0]
               pupil_vals_y[x_i, y_i] = e_vec[1]
               pupil_vals_mag[x_i, y_i] = e_mag

               x_list[i] = pupil_phi_range[y_i]
               y_list[i] = pupil_sin_theta_range[x_i]
               i += 1

     #grid of values, interpolated a bit more?

     grid_x, grid_y = np.meshgrid(x_range, y_range)
     points = (x_list, y_list)

     printif('##', show_prints)

     vals_efield_x = pupil_vals_x.flatten()
     vals_efield_y = pupil_vals_y.flatten()

     vals_intensity_x = (vals_efield_x*np.conjugate(vals_efield_x)).real
     vals_intensity_y = (vals_efield_y*np.conjugate(vals_efield_y)).real

     data_x = griddata(points, pupil_vals_x.flatten(), (grid_x, grid_y),\
          method='cubic',fill_value=0)
     data_y = griddata(points, pupil_vals_y.flatten(), (grid_x, grid_y),\
          method='cubic',fill_value=0)

     data_intensity_x = griddata(points, vals_intensity_x, (grid_x, grid_y),\
          method='cubic',fill_value=0)
     data_intensity_y = griddata(points, vals_intensity_y, (grid_x, grid_y),\
          method='cubic',fill_value=0)

     if plot_pupil: 
          printif("Plotting...", show_prints)
          plot_single_dipole_cartesian((x_range,y_range),\
               (data_intensity_x, data_intensity_y),\
               (single_dipole.phi_d, single_dipole.alpha_d), save_dir)

     return points, vals_efield_x, vals_efield_y