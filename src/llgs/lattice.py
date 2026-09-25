import matplotlib.colors as mplcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon


def normalize(spins):
    return spins / np.linalg.norm(spins, axis=1, keepdims=True)


class lattice_2D:
    def __init__(self, n_a, n_b, n_site, r_a=None, r_b=None, r_site=None):
        """Create a two-dimensional spin lattice with optional geometry."""
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
            self.positions = (
                a[:, None] * self.r_a
                + b[:, None] * self.r_b
                + self.r_site[site]
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

    def plot(self, arrowscale=0.3, annotate_idx=False, draw_unitcell=False):
        x = self.positions[:, 0]
        y = self.positions[:, 1]
        sx = self.spins[:, 0]
        sy = self.spins[:, 1]
        sz = self.spins[:, 2]

        plt.style.use("dark_background")
        plt.set_cmap("bwr")
        fig, ax = plt.subplots()
        cmap = plt.get_cmap("bwr")
        norm = mplcolors.Normalize(vmin=-1, vmax=1)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        plt.colorbar(sm, ax=ax)
        ax.scatter(x, y, c="yellow", s=2)
        if np.linalg.norm(sx) == np.linalg.norm(sy) == 0 and np.linalg.norm(sz) != 0:
            ax.scatter(x, y, c=sz, s=2, norm=norm)
        elif np.linalg.norm(sx) != 0 or np.linalg.norm(sy) != 0:
            ax.quiver(x, y, sx * arrowscale, sy * arrowscale, sz, norm=norm)
        if annotate_idx:
            for i, (x_pos, y_pos) in enumerate(zip(x, y)):
                ax.annotate(
                    str(i),
                    (x_pos, y_pos),
                    xycoords="data",
                    xytext=(1.5, 1.5),
                    textcoords="offset points",
                )

        if draw_unitcell:
            if not self.has_geometry:
                raise ValueError("geometry is required to draw the unit cell")
            origin = np.zeros(2)
            v3 = self.r_a + self.r_b
            unit_cell = Polygon(
                [origin, self.r_a, v3, self.r_b],
                closed=True,
                edgecolor="blue",
                facecolor="lightblue",
                alpha=0.5,
            )
            ax.add_patch(unit_cell)

        ax.set_aspect("equal")
        ax.axis("off")
