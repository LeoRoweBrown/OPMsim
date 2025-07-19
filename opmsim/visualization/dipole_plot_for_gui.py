from matplotlib.figure import Figure
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt
from matplotlib.pyplot import quiver
import numpy as np

class plot_dipole_source_3d(Figure):
    """Dipole 3d scatter to show orientations, subclass of mpl Figure for direct widget integration"""
    def __init__(self, alpha, phi, alphas=(), directional_arrow=None,
                 show_plot=True, dipole_style='arrow', **kwargs):
        """Create the dipole plot in the construtor.

        Args:
            alpha (ndarray): alpha_d angles, polar angle from z axis
            phi (ndarray): phi_d azimuthal angles, measured from x axis
            alphas (tuple, optional): opacity of points/arrows. Defaults to ().
            directional_arrow (ndarray, optional): 1x3 array to draw an excitation arrow. Defaults to None.
            show_plot (bool, optional): whether to show plot with plt.show(). Defaults to True.
            dipole_style (str, optional): Draw dipoles as points or double-ended arrows. Defaults to 'arrow'.
        """
        super().__init__(**kwargs)

        # convert from the polar coordinates to Cartesian
        xs = np.cos(alpha) * np.cos(phi)
        ys = np.cos(alpha) * np.sin(phi)
        zs = np.sin(alpha)
        alphas = np.atleast_1d(np.squeeze(alphas))  # from n x 1 x 1 to n
        n_dipoles = len(phi)
        origin = np.zeros(n_dipoles)

        if alphas.size == n_dipoles:
            color = [(0, 0, 1, alpha) for alpha in alphas]
        else:
            color = (0, 0, 1, 1)

        self.ax = self.add_subplot(projection='3d')
        origin = np.zeros_like(xs)
        if dipole_style == 'arrow':
            lw = 1 / np.log2(n_dipoles + 1)  # scale linewidth inversely with number of dipoles
            self.ax.quiver(origin, origin, origin, xs, ys, zs, color=color, lw=lw, arrow_length_ratio=0.1)
            if dipole_style == 'double_arrow':
                self.ax.quiver(origin, origin, origin, -xs, -ys, -zs, color=color, lw=lw, arrow_length_ratio=0.1)
        elif dipole_style == 'points':
            self.ax.scatter(xs, ys, zs, marker='o', s=5, c=color)
        elif dipole_style == 'points_both_ends':
            self.ax.scatter(xs, ys, zs, marker='o', s=5, c=color, label='+ve dipole end')
            self.ax.scatter(-xs, -ys, -zs, marker='x', s=5, c=color, linewidths=1, label='-ve dipole end')
            self.ax.legend()
        else:
            raise Exception("Invalid dipole_style {dipole_style}. Options: arrow, points, points_both_ends")
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        self.ax.set_zlabel("z")
        self.ax.set_xlim3d((-1, 1))
        self.ax.set_ylim3d((-1, 1))
        self.ax.set_zlim3d((-1, 1))
        self.ax.set_box_aspect([1, 1, 1])
        # self.ax.set_aspect('equal', adjustable='box')

        if directional_arrow is not None:
            phi_exc, alpha_exc = directional_arrow
            x_ex = np.cos(alpha_exc) * np.cos(phi_exc)
            y_ex = np.cos(alpha_exc) * np.sin(phi_exc)
            z_ex = np.sin(alpha_exc)  # unused at moment
            self.ax.quiver(
                0, 0, 0,
                x_ex, y_ex, z_ex,
                lw=3, scale=0.75, color='red')
        if show_plot:
            dummy = plt.figure()
            self.set_canvas(dummy.canvas)
            plt.show()

    def get_pyplot_fig(self):
        fig, ax = plt.subplots(FigureClass=self)