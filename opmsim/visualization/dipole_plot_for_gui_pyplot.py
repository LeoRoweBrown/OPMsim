
import matplotlib.pyplot as plt
import numpy as np

def plot_dipole_source_3d(alpha, phi, alphas=(), directional_arrow=None,
                          show_plot=True, dipole_style='arrow', **kwargs):
    """
    "Functional" version of the dipole plot. Returns the fig and axis objects, usable in
    jupyter notebook etc. unlike the class version in dipole_plot_for_gui which subclasses Figure.

    Args:
        alpha (ndarray): alpha_d angles, polar angle from z axis
        phi (ndarray): phi_d azimuthal angles, measured from x axis
        alphas (tuple, optional): opacity of points/arrows. Defaults to ().
        directional_arrow (ndarray, optional): 1x3 array to draw an excitation arrow. Defaults to None.
        show_plot (bool, optional): whether to show plot with plt.show(). Defaults to True.
        dipole_style (str, optional): Draw dipoles as points or double-ended arrows. Defaults to 'arrow'.
    Returns:
        (ax, fig) (tuple): axis and figure objects
    """
    fig = plt.figure(kwargs=kwargs)

    # convert from the polar coordinates to Cartesian
    xs = np.cos(alpha) * np.cos(phi)
    ys = np.cos(alpha) * np.sin(phi)
    zs = np.sin(alpha)
    n_dipoles = len(phi)
    origin = np.zeros(n_dipoles)
    alphas = np.atleast_1d(np.squeeze(alphas))  # from n x 1 x 1 to n
    if alphas.size == n_dipoles:
        color = [(0, 0, 1, alpha) for alpha in alphas]
    else:
        color = (0, 0, 1, 1)

    ax = fig.add_subplot(projection='3d')
    origin = np.zeros_like(xs)
    if dipole_style == 'arrow':
        lw = 1 / np.log2(n_dipoles + 1)  # scale linewidth inversely with number of dipoles
        ax.quiver(origin, origin, origin, xs, ys, zs, color=color, lw=lw)
        ax.quiver(origin, origin, origin, -xs, -ys, -zs, color=color, lw=lw)
    elif dipole_style == 'points':
        ax.scatter(xs, ys, zs, marker='o', s=5, c=color)
    elif dipole_style == 'points_both_ends':
        ax.scatter(xs, ys, zs, marker='o', s=5, c=color, label='+ve dipole end')
        ax.scatter(-xs, -ys, -zs, marker='x', s=5, c=color, linewidths=1, label='-ve dipole end')
        ax.legend()
    else:
        raise Exception("Invalid dipole_style {dipole_style}. Options: arrow, points, points_both_ends")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_xlim3d((-1, 1))
    ax.set_ylim3d((-1, 1))
    ax.set_zlim3d((-1, 1))
    # ax.set_aspect('equal', adjustable='box')

    if directional_arrow is not None:
        phi_exc, alpha_exc = directional_arrow
        x_ex = np.cos(alpha_exc) * np.cos(phi_exc)
        y_ex = np.cos(alpha_exc) * np.sin(phi_exc)
        z_ex = np.sin(alpha_exc)  # unused at moment
        ax.quiver(
            0, 0, 0,
            x_ex, y_ex, z_ex,
            lw=3, scale=0.75, color='red')
    if show_plot:
        dummy = plt.figure()
        fig.set_canvas(dummy.canvas)
        plt.show()

    return (ax, fig)
