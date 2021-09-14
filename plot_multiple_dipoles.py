import numpy as np
import dipole_source
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import os


def plot_random_dipole_source(NA, dipole_count, save_dir=None,
    file_name='multi_dipole_pupil.png', show_prints=False):
    dipoles = dipole_source.DipoleSource()
    dipoles.generate_dipoles(dipole_count)
    angles, data_intensity_x, data_intensity_y = dipoles.calculate_pupil_radiation(NA)

    title = "Radiation distribution in pupil for %d randomly distributed dipoles"\
        % (dipole_count)
    file_name = 'multi_dipole_pupil_N_%d.png' % dipole_count
    plot_pupil(title, angles, data_intensity_x, data_intensity_y, save_dir, file_name)
 
def plot_aligned_source(NA, theta, phi, save_dir=None, dipole_count=1):
    dipoles = dipole_source.DipoleSource(dipole_count)
    dipoles.add_dipoles(theta, phi, dipole_count)
    angles, data_intensity_x, data_intensity_y = dipoles.calculate_pupil_radiation(NA)

    phi_deg = phi*180/np.pi
    theta_deg = theta*180/np.pi

    title = "Radiation distribution in pupil for %d dipoles at "\
        + "$\\phi = %.1f^\circ, \\theta_d = %.1f^\circ$" %\
            (dipole_count, phi_deg, theta_deg)
    save_dir = os.path.join(save_dir, ("theta_d_%.1f" % theta_deg))
    file_name = "$phi%.1f_theta_d%.1f" % (phi_deg, theta_deg)

    plot_pupil(title, angles, data_intensity_x, data_intensity_y, save_dir, file_name)
    
def plot_pupil(title, angles, data_intensity_x, data_intensity_y, save_dir=None,
    file_name=None, show_prints=False):

    pupil_phi_range, pupil_sin_theta_range = angles
    print("Sin theta:",np.min(pupil_sin_theta_range), np.max(pupil_sin_theta_range))
    print("Phi:", np.min(pupil_phi_range), np.max(pupil_phi_range))

    data_intensity_sum = data_intensity_x + data_intensity_y
    print(data_intensity_y)

    max_for_scale = (np.max(data_intensity_y+data_intensity_x))

    fig = plt.figure(figsize=[10,4])
    # print(pupil_phi_range.size)
    # print(pupil_sin_theta_range.size)

    #Create a polar projection
    ax1 = fig.add_subplot(131, projection="polar")
    # pc1 = ax1.pcolormesh(pupil_phi_range,pupil_sin_theta_range,data_intensity_x.T,\
    pc1 = ax1.pcolormesh(pupil_phi_range,pupil_sin_theta_range,data_intensity_x,\
         shading='auto', vmin=0,vmax=max_for_scale)

    fig.colorbar(pc1, ax=ax1, fraction=0.05, pad=0.16)


    ax1.set_title("X polarisation pupil field distribution")
    # fix lims

    #Create a polar projection
    ax2 = fig.add_subplot(132, projection="polar")

    pc2 = ax2.pcolormesh(pupil_phi_range,pupil_sin_theta_range,data_intensity_y,\
         shading='auto', vmin=0,vmax=max_for_scale)

    fig.colorbar(pc2, ax=ax2, fraction=0.05, pad=0.16)

    ax2.set_title("Y polarisation pupil field distribution")

    ax3 = fig.add_subplot(133, projection="polar")

    pc2 = ax3.pcolormesh(pupil_phi_range,pupil_sin_theta_range,data_intensity_sum,\
         shading='auto', vmin=0,vmax=max_for_scale)
    fig.colorbar(pc2, ax=ax3, fraction=0.05, pad=0.16)

    ax3.set_title("X+Y polarisation pupil field distribution")

    # ax2.set_ylim([0, 1])
    fig.suptitle(title)

    fig.tight_layout()
    if save_dir is None:
        plt.show()
    else:
        save_dir = os.path.normpath(save_dir)
        print("Saving in %s" % save_dir)
        if not os.path.isdir(save_dir):
            print("No directory, %s, making it" % save_dir)
            os.makedirs(save_dir)
        full_save_path = os.path.join(save_dir, \
            file_name)
        fig.savefig(full_save_path, bbox_inches='tight')
    plt.close(fig)
    