from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib.pyplot import quiver
import numpy as np

class plot_dipole_source_3d(Figure):
    """Dipole 3d scatter to show orientations, subclass of mpl Figure"""
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
        
        self.ax = self.add_subplot(projection='3d')
        origin = np.zeros_like(xs)
        if dipole_style == 'arrow':
            self.ax.quiver(origin, origin, origin, xs, ys, zs)
            self.ax.quiver(origin, origin, origin, -xs, -ys, -zs)
        else:
            self.ax.scatter(xs, ys, zs)
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        self.ax.set_zlabel("z")
        self.ax.set_xlim3d((-1, 1))
        self.ax.set_ylim3d((-1, 1))
        self.ax.set_zlim3d((-1, 1))

        if directional_arrow is not None:
            phi_exc, alpha_exc = directional_arrow
            x_ex = np.cos(alpha_exc) * np.cos(phi_exc)
            y_ex = np.cos(alpha_exc) * np.sin(phi_exc)
            z_ex = np.sin(alpha_exc)  # unused at moment
            self.ax.quiver(
                0, 0, 0,
                x_ex, y_ex, z_ex,
                lw=5, color='red')
        if show_plot:
            plt.show()
