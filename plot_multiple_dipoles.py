import numpy as np
import dipole_source
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import os

def plot_single_dipole_source(dipole_alpha, dipole_phi, NA,
    dipole_obj=None, save_dir=None,
    show_prints=False, pupil='curved', rescale_energy=False):

    dipoles = dipole_source.DipoleSource()
    dipoles.add_dipoles(dipole_alpha, dipole_phi, 1)
    angles, data_intensity_x, data_intensity_y = \
        dipoles.calculate_pupil_radiation(NA, pupil=pupil,
            rescale_energy=rescale_energy)

    dipole_phi_deg = dipole_phi*180/np.pi
    dipole_alpha_deg = dipole_alpha*180/np.pi

    file_name = 'single_dipole_alpha_%.1f_phi_%.1f_%s_pupil.png' % \
        (dipole_alpha_deg, dipole_phi_deg, pupil)

    title = ("Single dipole radiation in pupil for "\
        + "$\\phi_d = %.1f^\circ, \\alpha_d = %.1f^\circ$ (%s pupil)" %\
            (dipole_phi_deg, dipole_alpha_deg, pupil))

    dipole_angles=[dipole_alpha, dipole_phi]

    plot_pupil(title, angles, data_intensity_x, data_intensity_y, save_dir,
        file_name, plot_arrow=dipole_angles)

def plot_random_dipole_source(NA, dipole_count, save_dir=None,
    file_name='multi_dipole_pupil.png', randomly=False, show_prints=False,
    pupil='curved', rescale_energy=False):

    dipoles = dipole_source.DipoleSource()
    dipoles.generate_dipoles(dipole_count, randomly=randomly)
    angles, data_intensity_x, data_intensity_y = dipoles.calculate_pupil_radiation(NA,
        pupil=pupil, rescale_energy=rescale_energy)

    title = "Radiation distribution in pupil for %d randomly distributed dipoles"\
        % (dipole_count)
    file_name = 'multi_dipole_pupil_N_%d.png' % dipole_count
    plot_pupil(title, angles, data_intensity_x, data_intensity_y, save_dir, file_name)
 
def plot_aligned_source(NA, dipole_alpha, dipole_phi, save_dir=None, dipole_count=1):
    dipoles = dipole_source.DipoleSource(dipole_count)
    dipoles.add_dipoles(dipole_alpha, dipole_phi, dipole_count)
    angles, data_intensity_x, data_intensity_y = dipoles.calculate_pupil_radiation(NA)

    phi_deg = dipole_phi*180/np.pi
    alpha_deg = dipole_alpha*180/np.pi

    title = "Radiation distribution in pupil for %d dipoles at "\
        + "$\\phi = %.1f^\circ, \\alpha_d = %.1f^\circ$" %\
            (dipole_count, dipole_alpha, alpha_deg)
    save_dir = os.path.join(save_dir, ("alpha_d_%.1f" % alpha_deg))
    file_name = "$phi%.1f_alpha_d%.1f" % (phi_deg, alpha_deg)

    plot_pupil(title, angles, data_intensity_x, data_intensity_y, save_dir, file_name)
    
def plot_pupil(title, angles, data_intensity_x, data_intensity_y, save_dir=None,
    file_name=None, show_prints=False, plot_arrow=None):

    dipole_alpha = None
    dipole_phi = None
    pupil_phi_range, pupil_sin_theta_range = angles
    print("Sin theta:",np.min(pupil_sin_theta_range), np.max(pupil_sin_theta_range))
    print("Phi:", np.min(pupil_phi_range), np.max(pupil_phi_range))

    data_intensity_sum = data_intensity_x + data_intensity_y
    # print(data_intensity_y)

    max_for_scale = (np.max(data_intensity_y+data_intensity_x))

    fig = plt.figure(figsize=[10,4])

    #Create a polar projection
    ax1 = fig.add_subplot(131, projection="polar")
    # pc1 = ax1.pcolormesh(pupil_phi_range,pupil_sin_theta_range,data_intensity_x.T,\
    pc1 = ax1.pcolormesh(pupil_phi_range,pupil_sin_theta_range,data_intensity_x,\
         shading='auto', vmin=0,vmax=max_for_scale)

    fig.colorbar(pc1, ax=ax1, fraction=0.05, pad=0.16)

    if plot_arrow is not None:
        dipole_alpha = plot_arrow[0]
        dipole_phi = plot_arrow[1]
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

    pc2 = ax2.pcolormesh(pupil_phi_range,pupil_sin_theta_range,data_intensity_y,\
         shading='auto', vmin=0,vmax=max_for_scale)

    fig.colorbar(pc2, ax=ax2, fraction=0.05, pad=0.16)

    if plot_arrow is not None:
        arrow_len = np.abs(0.5*np.cos(dipole_alpha))
        flip_phi = (dipole_phi+np.pi) % (2*np.pi)
        # p1 = patches.FancyArrowPatch((flip_phi, arrow_len), (dipole_phi, arrow_len), arrowstyle='<->',\
        #      mutation_scale=20)
        p2 = patches.FancyArrowPatch((flip_phi, arrow_len), (dipole_phi, arrow_len), arrowstyle='<->',\
            mutation_scale=20)
        ax2.add_patch(p2)

    ax2.set_title("Y polarisation pupil field distribution")

    ax3 = fig.add_subplot(133, projection="polar")

    pc2 = ax3.pcolormesh(pupil_phi_range,pupil_sin_theta_range,data_intensity_sum,\
         shading='auto', vmin=0,vmax=max_for_scale)
    fig.colorbar(pc2, ax=ax3, fraction=0.05, pad=0.16)

    if plot_arrow is not None:
        arrow_len = np.abs(0.5*np.cos(dipole_alpha))
        flip_phi = (dipole_phi+np.pi) % (2*np.pi)
        # p1 = patches.FancyArrowPatch((flip_phi, arrow_len), (dipole_phi, arrow_len), arrowstyle='<->',\
        #      mutation_scale=20)
        p3 = patches.FancyArrowPatch((flip_phi, arrow_len), (dipole_phi, arrow_len), arrowstyle='<->',\
            mutation_scale=20)
        ax3.add_patch(p3)

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
    