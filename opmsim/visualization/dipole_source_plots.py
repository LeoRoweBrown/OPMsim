import numpy as np
from matplotlib import pyplot as plt
from typing import Union
from numpy.typing import ArrayLike

def plot_dipole_source_3d(alpha, phi, alphas=(),
                          directional_arrow=None,
                          show_plot=True, dipole_style='arrow'):
    """Dipole 3d scatter to show orientations, pyplot version
        TODO: remove this and replace with dipole_plot_for_gui_pyplot version
    Args:
        alpha (ndarray): alpha_d angles, polar angle from z axis
        phi (ndarray): phi_d azimuthal angles, measured from x axis
        alphas (tuple, optional): opacity of points/arrows. Defaults to ().
        directional_arrow (ndarray, optional): 1x3 array to draw an excitation arrow. Defaults to None.
        show_plot (bool, optional): whether to show plot with plt.show(). Defaults to True.
        dipole_style (str, optional): Draw dipoles as points or double-ended arrows. Defaults to 'arrow'.
    """

    # convert from the polar coordinates to Cartesian
    xs = np.cos(alpha) * np.cos(phi)
    ys = np.cos(alpha) * np.sin(phi)
    zs = np.sin(alpha)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    n_dipoles = len(phi)
    origin = np.zeros(n_dipoles)
    if len(alphas) == n_dipoles:
        color = [('blue', alpha) for alpha in alphas]
    else:
        color = 'blue'
    if dipole_style == 'arrow':
        ax.quiver(origin, origin, origin, xs, ys, zs, color=color)
        ax.quiver(origin, origin, origin, -xs, -ys, -zs, color=color)
    else:
        ax.scatter(xs, ys, zs, c=color)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_xlim3d((-1, 1))
    ax.set_ylim3d((-1, 1))
    ax.set_zlim3d((-1, 1))

    if directional_arrow is not None:
        phi_exc, alpha_exc = directional_arrow
        x_ex = np.cos(alpha_exc) * np.cos(phi_exc)
        y_ex = np.cos(alpha_exc) * np.sin(phi_exc)
        z_ex = np.sin(alpha_exc)
        ax.quiver(
            0, 0, 0,
            x_ex, y_ex, z_ex,
            lw=5, color='red')
    if show_plot:
        plt.show()
    return fig

def plot_points_on_sphere(alpha, phi, alphas=(), style='arrow', directional_arrow=None, show_plot=True):
    """
    Plot points described by polar angle, alpha, and azimuthal angle, phi, on a sphere.
    Alpha measured from x.

    Args:
        alpha (_type_): _description_
        phi (_type_): _description_
        alphas (list, optional): Array of opacity, must have same length as alpha, phi or is scalar. Defaults to [1].
        style (str, optional): _description_. Defaults to 'arrow'.
        directional_arrow (_type_, optional): _description_. Defaults to None.
        show_plot (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: _description_
    """
    # convert from the polar coordinates to Cartesian
    x = np.cos(alpha) * np.cos(phi)
    y = np.cos(alpha) * np.sin(phi)
    z = np.sin(alpha)

    # face and edge colors
    facec = np.asarray(np.tile([0, 0, 0, 0], [len(x), 1]), dtype=np.float64)
    edgec = np.asarray(np.tile([0, 0, 0, 1], [len(x), 1]), dtype=np.float64)
    alphas = np.squeeze(alphas)  # because it has a Nx1x1.. shape for broadcasting when multiplied by intensity
    if alphas.size == 0:
        alphas = np.ones_like(x)
    elif alphas.size == 1:
        alphas = np.ones_like(x) * alphas
    alphas /= np.max(alphas)  # avoid a rogue alpha > 1 (yes it happens). TODO replace with cap?
    origin = np.zeros_like(x)

    # plot to verify distribution
    f2d = plt.figure(figsize=(14, 7))
    ax_xz = f2d.add_subplot(131)

    edgec[:, 3] = alphas  # apply the opacity to the opacity element
    facec[:, 3] = alphas

    y_mask = y < 0  # get negative y to hide
    facec[:, 3] = alphas  # set alphas on face color
    facec[y_mask, 3] = 0

    arrow_c = [('black', alpha) for alpha in alphas]
    if len(arrow_c) == 1:
        arrow_c = arrow_c[0]

    ax_xz.set_title("Distribution of dipole points \n on sphere (ZX)", fontsize=18)
    ax_xz.set_xlabel("z", fontsize=16)
    ax_xz.set_ylabel("x", fontsize=16)
    ax_xz.set_aspect("equal")
    ax_xz.set_xlim(-1, 1)
    ax_xz.set_ylim(-1, 1)
    if style == 'arrow':
        ax_xz.quiver(origin, origin, z, x, scale=2, color=arrow_c,
                     lw=8)
    else:
        ax_xz.scatter(z, x, s=5, facecolors=facec, edgecolors=edgec)

    ax_xy = f2d.add_subplot(132)

    z_mask = z < 0  # hide negative z
    facec[:, 3] = alphas  # reset alphas on face color
    facec[z_mask, 3] = 0  # transparent

    ax_xy.set_title("Distribution of dipole points \n on sphere (XY)", fontsize=18)

    if style == 'arrow':
        ax_xy.quiver(origin, origin, x, y, scale=2, color=arrow_c)
    else:
        ax_xy.scatter(x, y, s=5, facecolors=facec, edgecolors=edgec)

    ax_xy.set_xlabel("x", fontsize=16)
    ax_xy.set_ylabel("y", fontsize=16)
    ax_xy.set_aspect("equal")
    ax_xy.set_xlim(-1, 1)
    ax_xy.set_ylim(-1, 1)

    ax_zy = f2d.add_subplot(133)

    x_mask = x < 0  # hide negative x
    facec[:, 3] = alphas  # reset alphas on face color
    facec[x_mask, 3] = 0  # transparent

    ax_zy.set_title("Distribution of dipole points \n on sphere (ZY)", fontsize=18)

    if style == 'arrow':
        ax_zy.quiver(origin, origin, z, y, scale=2, color=arrow_c)
    else:
        ax_zy.scatter(z, y, s=5, facecolors=facec, edgecolors=edgec)

    ax_zy.set_xlabel("z", fontsize=16)
    ax_zy.set_ylabel("y", fontsize=16)
    ax_zy.set_aspect("equal")
    ax_zy.set_xlim(-1, 1)
    ax_zy.set_ylim(-1, 1)
    f2d.tight_layout()

    if directional_arrow is not None:
        phi_exc, alpha_exc = directional_arrow
        print("plot exc arrow")
        x_ex = np.cos(alpha_exc) * np.cos(phi_exc)
        y_ex = np.cos(alpha_exc) * np.sin(phi_exc)
        z_ex = np.sin(alpha_exc)  # unused at moment

        w = 0.2
        if abs(x_ex) < 1e-9 and abs(z_ex) < 1e-9:
            ax_xz.plot(0, 0, marker="o", markersize=16, markeredgecolor="k", markeredgewidth=2)
        else:
            ax_xz.arrow(
                0, 0, (1 - w) * x_ex, (1 - w) * z_ex, head_width=0.2, head_length=0.2, linewidth=2
            )  # ,length_includes_head=True)
            ax_xz.arrow(
                0, 0, -(1 - w) * x_ex, -(1 - w) * z_ex, head_width=0.2, head_length=0.2, linewidth=2
            )  # ,length_includes_head=True)

        if abs(x_ex) < 1e-9 and abs(y_ex) < 1e-9:
            ax_xy.plot(0, 0, marker="o", markersize=16, markeredgecolor="k", markeredgewidth=2)
        else:
            ax_xy.arrow(
                0, 0, (1 - w) * x_ex, (1 - w) * y_ex, head_width=0.2, head_length=0.2, linewidth=2
            )  # ,length_includes_head=True)
            ax_xy.arrow(
                0, 0, -(1 - w) * x_ex, -(1 - w) * y_ex, head_width=0.2, head_length=0.2, linewidth=2
            )  # ,length_includes_head=True)
        if abs(z_ex) < 1e-9 and abs(y_ex) < 1e-9:
            ax_zy.plot(0, 0, marker="o", markersize=16, markeredgecolor="k", markeredgewidth=2)
        else:
            ax_zy.arrow(
                0, 0, (1 - w) * z_ex, (1 - w) * y_ex, head_width=0.2, head_length=0.2, linewidth=2
            )  # ,length_includes_head=True)
            ax_zy.arrow(
                0, 0, -(1 - w) * z_ex, -(1 - w) * y_ex, head_width=0.2, head_length=0.2, linewidth=2
            )  # ,length_includes_head=True)
    if show_plot:
        plt.show()
    return f2d

def plot_ray_sphere(phi, theta, plot_histo=False):
    x = np.sin(theta) * np.sin(phi)
    y = np.sin(theta) * np.cos(phi)
    z = np.cos(theta)
    zeros = np.zeros_like(x)

    # plot to verify distribution
    f2d = plt.figure(figsize=(14, 7))
    ax_xz = f2d.add_subplot(131)
    ax_xz.scatter(z, x, s=2)
    ax_xz.set_title("Distribution of ray points \n on sphere (ZX)", fontsize=18)
    ax_xz.set_xlabel("z", fontsize=16)
    ax_xz.set_ylabel("x", fontsize=16)
    ax_xz.set_aspect("equal")
    # Stacking to vectorize the lines drawn from origin
    ax_xz.plot(
        np.vstack([np.zeros_like(z), z]),
        np.vstack([np.zeros_like(x), x]),
        color=[0, 0, 0, 0.15]
    )
    ax_xy = f2d.add_subplot(132)
    ax_xy.set_title("Distribution of ray points \n on sphere (YX)", fontsize=18)
    ax_xy.scatter(y, x, s=2)
    ax_xy.set_xlabel("y", fontsize=16)
    ax_xy.set_ylabel("x", fontsize=16)
    ax_xy.set_aspect("equal")
    ax_xy.plot(
        np.vstack([np.zeros_like(y), y]),
        np.vstack([np.zeros_like(x), x]),
        color=[0, 0, 0, 0.15]
    )
    ax_zy = f2d.add_subplot(133)
    ax_zy.set_title("Distribution of ray points \n on sphere (YZ)", fontsize=18)
    ax_zy.scatter(z, y, s=2)
    ax_zy.set_xlabel("z", fontsize=16)
    ax_zy.set_ylabel("y", fontsize=16)
    ax_zy.set_aspect("equal")
    ax_zy.plot(
        np.vstack([np.zeros_like(z), z]),
        np.vstack([np.zeros_like(y), y]),
        color=[0, 0, 0, 0.15]
    )
    f2d.tight_layout()
    plt.show()

    # also plot histos
    if plot_histo:
        sort_phi = list(set(phi))
        sort_phi.sort()
        print("max phi", np.max(phi))
        print("phi sep", sort_phi[1] - sort_phi[0])
        plt.hist(phi, bins=50)
        plt.xlabel(phi)
        plt.show()


def display_pupil_rays(rays):
    fig = plt.figure(figsize=[8, 8])
    ax = fig.add_subplot(projection="polar")
    c = ax.scatter(rays.phi, np.sin(rays.theta), s=1)
    ax.set_ylim(0, 1)  # show whole NA=1 pupil
    ax.set_title("Simulated ray distribution in pupil (sine projection)")
    plt.show()
