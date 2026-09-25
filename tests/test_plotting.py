import matplotlib.animation as mpl_animation
import matplotlib.pyplot as plt
import numpy as np
import pytest

from llgs import ReadResult


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

    output = result.animate(period=1, save_fn=filename, fps=12)

    assert isinstance(output, mpl_animation.FuncAnimation)
    assert saved == {"filename": filename, "fps": 12}
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
