import matplotlib.animation as mpl_animation
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pytest

from llgs import Lattice_2D, ReadResult


STYLE_KEYS = ("axes.facecolor", "figure.facecolor", "image.cmap", "text.color")


def current_style():
    return {key: mpl.rcParams[key] for key in STYLE_KEYS}


def test_lattice_plot_does_not_change_global_style():
    lattice = Lattice_2D(n_a=1, n_b=1, n_site=1)
    lattice.spins[:] = [1.0, 0.0, 0.0]
    before = current_style()

    lattice.plot(display=False)

    assert current_style() == before
    plt.close("all")


def test_lattice_plot_accepts_light_and_custom_themes():
    lattice = Lattice_2D(n_a=1, n_b=1, n_site=1)
    lattice.spins[:] = [1.0, 0.0, 0.0]

    figure, axes = lattice.plot(theme="light", display=False)
    assert axes.get_facecolor() == (1.0, 1.0, 1.0, 1.0)
    plt.close(figure)

    figure, axes = lattice.plot(
        theme={"lattice_color": "magenta"},
        display=False,
    )
    np.testing.assert_allclose(
        axes.collections[0].get_facecolor()[0],
        mpl.colors.to_rgba("magenta"),
    )
    plt.close(figure)

    with pytest.raises(ValueError, match="theme"):
        lattice.plot(theme="unknown", display=False)


def test_animate_saves_simulation_result(monkeypatch, tmp_path):
    result = ReadResult.__new__(ReadResult)
    result.structure = np.array(
        [[0, 0, 0, 0.0, 0.0], [0, 0, 1, 1.0, 0.0]]
    )
    result.spin_datas = np.array(
        [
            [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]],
            [[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]],
        ]
    )
    result.times = np.array([0.0, 0.1])
    saved = {}

    def fake_save(animation, filename, fps):
        animation._draw_was_started = True
        saved.update(filename=filename, fps=fps)

    monkeypatch.setattr(mpl_animation.FuncAnimation, "save", fake_save)
    filename = tmp_path / "spins.gif"
    before = current_style()

    output = result.animate(period=1, save_fn=filename, fps=12, display=False)

    assert isinstance(output, mpl_animation.FuncAnimation)
    assert saved == {"filename": filename, "fps": 12}
    assert current_style() == before
    plt.close("all")


def test_animate_requires_geometry_and_positive_period(tmp_path):
    result = ReadResult.__new__(ReadResult)
    result.structure = np.zeros((1, 3))
    result.spin_datas = np.zeros((1, 1, 3))
    result.times = np.zeros(1)
    with pytest.raises(ValueError, match="x and y"):
        result.animate(period=1, save_fn=tmp_path / "spins.gif")

    result.structure = np.zeros((1, 5))
    with pytest.raises(ValueError, match="positive"):
        result.animate(period=0, save_fn=tmp_path / "spins.gif")
