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
    pupil='curved'):

     if dipole_obj is None:
          single_dipole = dipole.Dipole(dipole_phi, dipole_alpha) # phi then theta
     elif isinstance(dipole_obj, dipole.Dipole):
          single_dipole = dipole_obj
     else:
          raise Exception("Invalid input for dipole_obj, should be instance of Dipole from dipole")
    