import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from scipy.interpolate import griddata
import dipole as dipole
import os

## plot polar plots

def plot_single_dipole(angles, data, dipole_angles, save_dir = None):
    pupil_phi_range, pupil_sin_theta_range = angles
    dipole_phi, dipole_alpha = dipole_angles
    data_intensity_x, data_intensity_y = data
    data_intensity_sum = data_intensity_x + data_intensity_y
    # dipole_theta_from_y = (dipole_theta - np.pi/2) % (2*np.pi)
    vmax = np.max(data_intensity_sum)

    fig = plt.figure(figsize=[10,4])
    #Create a polar projection
    ax1 = fig.add_subplot(131, projection="polar")
    # pc1 = ax1.pcolormesh(pupil_phi_range,pupil_sin_theta_range,data_x.T, shading='auto')
    print(pupil_phi_range.size)
    print(pupil_sin_theta_range.size)
    print(data_intensity_x.T.size)
    pc1 = ax1.pcolormesh(pupil_phi_range,pupil_sin_theta_range,data_intensity_x.T,\
         shading='auto', vmin=0,vmax=vmax)

    # fig.colorbar(pc1, ax=ax1, fraction=0.046, pad=0.06)
    fig.colorbar(pc1, ax=ax1, fraction=0.04, pad=0.16)

    # ax1.pcolormesh(pupil_phi_range,pupil_sin_theta_range,pupil_vals_x, shading='auto')

    # dipole arrow length depends on theta?
    arrow_len = np.abs(0.5*np.cos(dipole_alpha))

    flip_phi = (dipole_phi+np.pi) % (2*np.pi)
    # p1 = patches.FancyArrowPatch((flip_phi, arrow_len), (dipole_phi, arrow_len), arrowstyle='<->',\
    #      mutation_scale=20)
    p1 = patches.FancyArrowPatch((flip_phi, arrow_len), (dipole_phi, arrow_len), arrowstyle='<->',\
         mutation_scale=20)
    ax1.add_patch(p1)
    ax1.set_title("X polarisation pupil field distribution")
    # fix lims

    #Create a polar projection
    ax2 = fig.add_subplot(132, projection="polar")

    # pc2 = ax2.pcolormesh(pupil_phi_range,pupil_sin_theta_range,data_y.T, shading='auto')
    pc2 = ax2.pcolormesh(pupil_phi_range,pupil_sin_theta_range,data_intensity_y.T,\
         shading='auto', vmin=0,vmax=vmax)
    #fig.colorbar(pc2, ax=ax2, fraction=0.046, pad=0.06)
    fig.colorbar(pc2, ax=ax2, fraction=0.04, pad=0.16)
    p2 = patches.FancyArrowPatch((flip_phi, arrow_len), (dipole_phi, arrow_len),\
        arrowstyle='<->', mutation_scale=20)
    ax2.add_patch(p2)
    ax2.set_title("Y polarisation pupil field distribution")

    

    ax3 = fig.add_subplot(133, projection="polar")

    pc2 = ax3.pcolormesh(pupil_phi_range,pupil_sin_theta_range,data_intensity_sum.T,\
         shading='auto', vmin=0,vmax=vmax)
    fig.colorbar(pc2, ax=ax3, fraction=0.04, pad=0.16)
    p3 = patches.FancyArrowPatch((flip_phi, arrow_len), (dipole_phi, arrow_len),\
        arrowstyle='<->', mutation_scale=20)
    ax3.add_patch(p3)
    ax3.set_title("X+Y polarisation pupil field distribution")

    dipole_phi_deg = dipole_phi*180/np.pi
    dipole_alpha_deg = dipole_alpha*180/np.pi

    # ax2.set_ylim([0, 1])
    fig.suptitle("Single dipole radiation in pupil for "\
        + "$\\phi_d = %.1f^\circ, \\alpha_d = %.1f^\circ$" %\
            (dipole_phi_deg, dipole_alpha_deg))

    fig.tight_layout()
    if save_dir is None:
        plt.show()
    else:
        save_dir = os.path.normpath(save_dir)
        print("Saving in %s" % save_dir)
        save_dir_theta = os.path.join(save_dir, 'theta_dipole_%.1f' \
            % dipole_alpha_deg)
        if not os.path.isdir(save_dir_theta):
            print("No directory, %s, making it" % save_dir_theta)
            os.makedirs(save_dir_theta)
        full_save_path = os.path.join(save_dir_theta, \
            'single_dipole_alpha_%.1f_phi_%.1f.png' %\
            (dipole_alpha_deg, dipole_phi_deg))
        fig.savefig(full_save_path, bbox_inches='tight')
    plt.close(fig)

def plot_single_dipole_cartesian(xy, data, dipole_angles, save_dir = None):
    x_range, y_range = xy
    dipole_phi, dipole_alpha = dipole_angles
    data_intensity_x, data_intensity_y = data
    data_intensity_sum = data_intensity_x + data_intensity_y
    # dipole_theta_from_y = (dipole_theta - np.pi/2) % (2*np.pi)

    fig = plt.figure(figsize=[10,4])
    #Create a polar projection
    ax1 = fig.add_subplot(131)#, projection="polar")
    # pc1 = ax1.pcolormesh(pupil_phi_range,pupil_sin_theta_range,data_x.T, shading='auto')
    print(x_range.size)
    print(y_range.size)
    print(data_intensity_x.T.size)
    pc1 = ax1.pcolormesh(x_range,y_range,data_intensity_x.T,\
         shading='auto', vmin=0,vmax=1)

    # fig.colorbar(pc1, ax=ax1, fraction=0.046, pad=0.06)
    fig.colorbar(pc1, ax=ax1, fraction=0.05, pad=0.16)

    # ax1.pcolormesh(pupil_phi_range,pupil_sin_theta_range,pupil_vals_x, shading='auto')

    # dipole arrow length depends on theta?
    arrow_len = np.abs(0.5*np.cos(dipole_alpha))

    flip_phi = (dipole_phi+np.pi) % (2*np.pi)
    # p1 = patches.FancyArrowPatch((flip_phi, arrow_len), (dipole_phi, arrow_len), arrowstyle='<->',\
    #      mutation_scale=20)
    p1 = patches.FancyArrowPatch((flip_phi, arrow_len), (dipole_phi, arrow_len), arrowstyle='<->',\
         mutation_scale=20)
    ax1.add_patch(p1)
    ax1.set_title("X polarisation pupil field distribution")
    # fix lims

    #Create a polar projection
    ax2 = fig.add_subplot(132)#, projection="polar")

    # pc2 = ax2.pcolormesh(pupil_phi_range,pupil_sin_theta_range,data_y.T, shading='auto')
    pc2 = ax2.pcolormesh(x_range,y_range,data_intensity_y.T,\
         shading='auto', vmin=0,vmax=1)
    #fig.colorbar(pc2, ax=ax2, fraction=0.046, pad=0.06)
    fig.colorbar(pc2, ax=ax2, fraction=0.05, pad=0.16)
    p2 = patches.FancyArrowPatch((flip_phi, arrow_len), (dipole_phi, arrow_len),\
        arrowstyle='<->', mutation_scale=20)
    ax2.add_patch(p2)
    ax2.set_title("Y polarisation pupil field distribution")

    

    ax3 = fig.add_subplot(133)#, projection="polar")

    pc2 = ax3.pcolormesh(x_range,y_range,data_intensity_sum.T,\
         shading='auto', vmin=0,vmax=1)
    fig.colorbar(pc2, ax=ax3, fraction=0.05, pad=0.16)
    p3 = patches.FancyArrowPatch((flip_phi, arrow_len), (dipole_phi, arrow_len),\
        arrowstyle='<->', mutation_scale=20)
    ax3.add_patch(p3)
    ax3.set_title("X+Y polarisation pupil field distribution")

    dipole_phi_deg = dipole_phi*180/np.pi
    dipole_alpha_deg = dipole_alpha*180/np.pi

    # ax2.set_ylim([0, 1])
    fig.suptitle("Single dipole radiation in pupil for "\
        + "$\\phi_d = %.1f^\circ, \\alpha_d = %.1f^\circ$" %\
            (dipole_phi_deg, dipole_alpha_deg))

    fig.tight_layout()
    if save_dir is None:
        plt.show()
    else:
        save_dir = os.path.normpath(save_dir)
        print("Saving in %s" % save_dir)
        save_dir_theta = os.path.join(save_dir, 'theta_dipole_%.1f' \
            % dipole_alpha_deg)
        if not os.path.isdir(save_dir_theta):
            print("No directory, %s, making it" % save_dir_theta)
            os.makedirs(save_dir_theta)
        full_save_path = os.path.join(save_dir_theta, \
            'single_dipole_alpha_%.1f_phi_%.1f.png' %\
            (dipole_alpha_deg, dipole_phi_deg))
        fig.savefig(full_save_path, bbox_inches='tight')
    plt.close(fig)