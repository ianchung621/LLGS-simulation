"""Visualization helpers for lattices and simulation results."""

from pathlib import Path
from typing import TYPE_CHECKING, Tuple, Union

import matplotlib.animation as animation
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Polygon

if TYPE_CHECKING:
    from .lattice import Lattice_2D
    from .read_results import ReadResult


def _plot_lattice(
    lattice: "Lattice_2D",
    arrowscale: float = 0.3,
    annotate_idx: bool = False,
    draw_unitcell: bool = False,
) -> Tuple[Figure, Axes]:
    """Plot lattice positions and the current spin configuration."""
    x = lattice.positions[:, 0]
    y = lattice.positions[:, 1]
    sx = lattice.spins[:, 0]
    sy = lattice.spins[:, 1]
    sz = lattice.spins[:, 2]

    plt.style.use("dark_background")
    plt.set_cmap("bwr")
    fig, ax = plt.subplots()
    cmap = plt.get_cmap("bwr")
    norm = colors.Normalize(vmin=-1, vmax=1)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    plt.colorbar(sm, ax=ax)
    ax.scatter(x, y, c="yellow", s=2)
    if np.linalg.norm(sx) == np.linalg.norm(sy) == 0 and np.linalg.norm(sz) != 0:
        ax.scatter(x, y, c=sz, s=2, norm=norm)
    elif np.linalg.norm(sx) != 0 or np.linalg.norm(sy) != 0:
        ax.quiver(x, y, sx * arrowscale, sy * arrowscale, sz, norm=norm)
    if annotate_idx:
        for index, (x_position, y_position) in enumerate(zip(x, y)):
            ax.annotate(
                str(index),
                (x_position, y_position),
                xycoords="data",
                xytext=(1.5, 1.5),
                textcoords="offset points",
            )

    if draw_unitcell:
        if not lattice.has_geometry:
            raise ValueError("geometry is required to draw the unit cell")
        origin = np.zeros(2)
        opposite_corner = lattice.r_a + lattice.r_b
        unit_cell = Polygon(
            [origin, lattice.r_a, opposite_corner, lattice.r_b],
            closed=True,
            edgecolor="blue",
            facecolor="lightblue",
            alpha=0.5,
        )
        ax.add_patch(unit_cell)

    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


def _animate(
    result: "ReadResult",
    period: int,
    save_fn: Union[str, Path],
    fps: int = 10,
) -> animation.FuncAnimation:
    """Animate saved spin data and write it to a GIF or video file."""
    if result.structure.shape[1] < 5:
        raise ValueError("structure must include x and y coordinates")
    if period <= 0:
        raise ValueError("period must be positive")

    plt.style.use("dark_background")
    plt.set_cmap("bwr")

    fig, ax = plt.subplots()
    line = ax.plot([], [])
    cmap = plt.get_cmap("bwr")
    norm = colors.Normalize(vmin=-1, vmax=1)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    plt.colorbar(sm, ax=ax)

    def initialize():
        return line

    def update_lattice(frame):
        frame = min(frame * period, len(result.spin_datas) - 1)
        x, y = result.structure[:, 3], result.structure[:, 4]
        sx, sy, sz = result.spin_datas[frame].T

        ax.clear()
        ax.scatter(x, y, c="yellow", s=1)
        ax.quiver(x, y, sx * 0.3, sy * 0.3, sz, norm=norm)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(f"time = {result.times[frame]:0.2f} ps")
        return line

    frame_count = max(1, len(result.times) // period)
    output = animation.FuncAnimation(
        fig,
        update_lattice,
        frames=frame_count,
        init_func=initialize,
        blit=True,
    )
    output.save(save_fn, fps=fps)
    return output
