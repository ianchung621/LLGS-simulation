from typing import Literal, Union

import numpy as np


def normalize(spins):
    return spins / np.linalg.norm(spins, axis=1, keepdims=True)


class Lattice_2D:
    def __init__(self, n_a, n_b, n_site, r_a=None, r_b=None, r_site=None):
        """Create a two-dimensional spin lattice.

        Parameters
        ----------
        n_a, n_b : int
            Number of unit cells along the two lattice vectors.
        n_site : int
            Number of sites in each unit cell.
        r_a, r_b : array-like of shape (2,), optional
            Cartesian lattice vectors. Geometry is omitted when these and
            ``r_site`` are all ``None``.
        r_site : array-like of shape (n_site, 2), optional
            Fractional site coordinates within a unit cell. A row ``(u, v)``
            places that site at ``u * r_a + v * r_b`` relative to the cell
            origin. It must be supplied together with ``r_a`` and ``r_b``.

        Notes
        -----
        The public arrays ``tags``, ``positions``, ``spins``,
        ``spin_velocities``, and ``structure`` contain the lattice state.
        ``structure`` has columns ``a, b, site`` and, when geometry is
        provided, the Cartesian columns ``x, y``.
        """
        geometry = (r_a, r_b, r_site)
        if any(value is not None for value in geometry) and not all(
            value is not None for value in geometry
        ):
            raise ValueError("r_a, r_b, and r_site must be provided together")

        self.n_site = n_site
        self.n_a = n_a
        self.n_b = n_b
        self.N = n_site * n_a * n_b

        a, b, site = np.meshgrid(
            np.arange(n_a),
            np.arange(n_b),
            np.arange(n_site),
            indexing="ij",
        )
        self.tags = np.stack(
            (a.ravel(), b.ravel(), site.ravel()), axis=1
        ).astype(int)
        self.positions = np.zeros((self.N, 2))
        self.spins = np.zeros((self.N, 3))
        self.spin_velocities = np.zeros((self.N, 3))

        self.r_a = None
        self.r_b = None
        self.r_site = None
        self.has_geometry = all(value is not None for value in geometry)

        if self.has_geometry:
            self.r_a = np.asarray(r_a)
            self.r_b = np.asarray(r_b)
            self.r_site = np.asarray(r_site)
            if self.r_a.shape != (2,):
                raise ValueError(
                    f"r_a must be a (2,) array, but got shape {self.r_a.shape}."
                )
            if self.r_b.shape != (2,):
                raise ValueError(
                    f"r_b must be a (2,) array, but got shape {self.r_b.shape}."
                )
            if self.r_site.shape != (self.n_site, 2):
                raise ValueError(
                    f"r_site must be a ({self.n_site}, 2) array, "
                    f"but got shape {self.r_site.shape}."
                )

            a, b, site = self.tags.T
            site_coordinates = self.r_site[site]
            self.positions = (
                a[:, None] * self.r_a
                + b[:, None] * self.r_b
                + site_coordinates[:, 0, None] * self.r_a
                + site_coordinates[:, 1, None] * self.r_b
            )
            self.structure = np.column_stack((self.tags, self.positions))
        else:
            self.structure = self.tags.copy()

    def output_lattice_structure(self, fn):
        """Write lattice data to a CSV file."""
        output = np.column_stack((np.arange(self.N), self.structure))
        columns = ["particle idx", "a", "b", "site"]
        formats = ["%d"] * 4
        if self.has_geometry:
            columns.extend(("x", "y"))
            formats.extend(("%.18e", "%.18e"))
        np.savetxt(
            fn,
            output,
            delimiter=",",
            header=",".join(columns),
            comments="",
            fmt=formats,
        )

    def initialize_spin(self, condition_dict, perturb=0.0):
        """Initialize spins from NumPy expressions over lattice coordinates."""
        condition_values = {
            "a": self.tags[:, 0],
            "b": self.tags[:, 1],
            "site": self.tags[:, 2],
        }
        if self.has_geometry:
            condition_values.update(
                {
                    "x": self.positions[:, 0],
                    "y": self.positions[:, 1],
                }
            )

        for cond_str, spin in condition_dict.items():
            cond = np.asarray(
                eval(cond_str, {"__builtins__": {}}, condition_values),
                dtype=bool,
            )
            if cond.shape != (self.N,):
                raise ValueError(
                    f"condition must produce shape ({self.N},), got {cond.shape}"
                )
            self.spins[cond] = spin

        self.spins += perturb * np.random.normal(0, 1, (self.N, 3))
        self.spins = normalize(self.spins)

    def plot(
        self,
        arrowscale=0.3,
        annotate_idx=False,
        draw_unitcell=False,
        display=True,
        theme: Union[Literal["light", "dark"], dict] = "dark",
    ):
        """Plot this lattice and its current spin configuration.

        Parameters
        ----------
        arrowscale : float, default 0.3
            Scale applied to in-plane spin arrows.
        annotate_idx : bool, default False
            Label each lattice site with its particle index.
        draw_unitcell : bool, default False
            Draw the unit-cell boundary. Geometry must be available.
        display : bool, default True
            Show the figure with Matplotlib when true.
        theme : {"light", "dark"} or dict, default "dark"
            Built-in theme name or custom values overriding ``DARK_THEME``.

        Returns
        -------
        tuple
            Matplotlib ``(figure, axes)`` objects.
        """
        from .plotting import _plot_lattice

        return _plot_lattice(
            self,
            arrowscale=arrowscale,
            annotate_idx=annotate_idx,
            draw_unitcell=draw_unitcell,
            display=display,
            theme=theme,
        )
