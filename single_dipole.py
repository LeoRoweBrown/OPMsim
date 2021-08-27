
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from scipy.interpolate import griddata
import dipole as dipole
from plot_single_dipole import plot_single_dipole
from tools import printif  # is the name tools a little dangerous? conflicts etc

def simulate_single_dipole(dipole_phi, dipole_theta, plot_pupil=None\
     save_dir=None, show_prints=False):
     single_dipole = dipole.dipole(dipole_phi, dipole_theta) # phi then theta

     pupil_sin_theta_range = np.linspace(0,1,50)
     pupil_theta_range = np.arcsin(pupil_sin_theta_range)
     pupil_phi_range = np.linspace(0,2*np.pi,100)

     printif('dipole angle: phi = %f theta = %f' % (single_dipole.phi*(180/np.pi),\
          single_dipole.theta_dipole_coords*(180/np.pi)), show_prints)

     pupil_vals_x = np.zeros([len(pupil_theta_range), len(pupil_phi_range)])
     pupil_vals_y = np.zeros([len(pupil_theta_range), len(pupil_phi_range)])

     phi_list = np.zeros([len(pupil_theta_range) * len(pupil_phi_range)])
     sin_theta_list = np.zeros([len(pupil_theta_range) * len(pupil_phi_range)])

     # evaluate the field across the pupil
     i = 0
     for t_i, theta in enumerate(pupil_theta_range):
          for p_i, phi in enumerate(pupil_phi_range):
               e_vec, e_mag = single_dipole.getEfield(theta, phi, 1)
               # print(e_vec)
               # print('######')
               pupil_vals_x[t_i, p_i] = e_vec[0]
               pupil_vals_y[t_i, p_i] = e_vec[1]

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
               (single_dipole.phi, single_dipole.theta), save_dir)
