import importlib

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pytest

from llgs.LLGS_simulation import LLGS_Simulation_2D
from llgs.lattice import lattice_2D
from llgs.read_results import ReadResult
from param.NiPS3 import NiPS3_params


simulation_module = importlib.import_module("llgs.LLGS_simulation")


def make_honeycomb(n_a=3, n_b=2):
    return lattice_2D(
        n_a=n_a,
        n_b=n_b,
        n_site=2,
        r_a=np.array([np.sqrt(3), 0]),
        r_b=np.array([0.5 * np.sqrt(3), 1.5]),
        r_site=np.array(
            [
                [0.5 * np.sqrt(3), 0.5],
                [np.sqrt(3), 1],
            ]
        ),
    )


def make_exchange_field(honeycomb, J_1=0, J_2=0, J_3=0):
    distances = np.linalg.norm(
        honeycomb.positions[:, None, :] - honeycomb.positions[None, :, :],
        axis=-1,
    )
    exchange = np.zeros(distances.shape)
    exchange[np.isclose(distances, 1, atol=1e-3, rtol=0)] = J_1 / 2
    exchange[np.isclose(distances, np.sqrt(3), atol=1e-3, rtol=0)] = J_2 / 2
    exchange[np.isclose(distances, 2, atol=1e-3, rtol=0)] = J_3 / 2
    return exchange


def test_lattice_constructor_requires_complete_geometry():
    lattice = lattice_2D(n_a=2, n_b=1, n_site=1)

    assert lattice.positions.shape == (2, 2)
    assert lattice.structure.shape == (2, 3)
    assert not lattice.has_geometry

    with pytest.raises(ValueError, match="must be provided together"):
        lattice_2D(
            n_a=2,
            n_b=1,
            n_site=1,
            r_a=np.array([1, 0]),
        )


def test_honeycomb_lattice_and_zigzag_initialization(tmp_path):
    honeycomb = make_honeycomb()

    assert honeycomb.N == 12
    assert honeycomb.tags.shape == (12, 3)
    assert honeycomb.positions.shape == (12, 2)
    assert honeycomb.structure.shape == (12, 5)
    np.testing.assert_allclose(
        honeycomb.positions[:2],
        [[0.5 * np.sqrt(3), 0.5], [np.sqrt(3), 1]],
    )

    honeycomb.initialize_spin(
        {
            "b % 2 == 0": np.array([1, 0, 0]),
            "b % 2 == 1": np.array([-1, 0, 0]),
        }
    )
    expected_x = np.where(honeycomb.tags[:, 1] % 2 == 0, 1, -1)
    np.testing.assert_array_equal(honeycomb.spins[:, 0], expected_x)
    np.testing.assert_array_equal(honeycomb.spins[:, 1:], 0)

    output_path = tmp_path / "honeycomb.csv"
    honeycomb.output_lattice_structure(output_path)
    assert output_path.read_text().splitlines()[0] == "particle idx,a,b,site,x,y"

    honeycomb.plot(draw_unitcell=True)
    assert plt.get_fignums()
    plt.close("all")


def test_fields_are_normalized_when_evolution_starts(tmp_path, monkeypatch):
    lattice = lattice_2D(n_a=1, n_b=1, n_site=2)
    lattice.spins[:] = [1, 0, 0]
    with pytest.raises(ValueError, match="H_DMI must have shape"):
        LLGS_Simulation_2D(lattice, H_DMI=np.zeros((2, 2, 2)))

    raw_exchange = np.array([[0.0, 4.0], [2.0, 0.0]])
    raw_dmi = np.zeros((3, 2, 2))
    raw_dmi[0] = [[0.0, 4.0], [2.0, 0.0]]
    captured = {}

    def capture_step(**fields):
        if not captured:
            captured.update(fields)
        return fields["spins"].copy(), fields["svels"].copy()

    monkeypatch.setattr(simulation_module, "_get_next_spin_euler", capture_step)
    simulation = LLGS_Simulation_2D(
        lattice,
        H_E=raw_exchange,
        H_DMI=raw_dmi,
        method="Euler",
        io_foldername=tmp_path,
        io_screen=False,
    )
    simulation.evolve(max_iters=1000)

    np.testing.assert_array_equal(simulation.H_E, raw_exchange)
    np.testing.assert_array_equal(simulation.H_DMI, raw_dmi)
    np.testing.assert_allclose(captured["H_E"], [[0, 3], [3, 0]])
    np.testing.assert_allclose(captured["H_DMI"][0], [[0, 1], [-1, 0]])


def test_nips3_simulation_output_can_be_read(tmp_path):
    honeycomb = make_honeycomb(n_a=2, n_b=2)
    honeycomb.initialize_spin(
        {
            "b % 2 == 0": np.array([1, 0, 0]),
            "b % 2 == 1": np.array([-1, 0, 0]),
        }
    )
    initial_spins = honeycomb.spins.copy()

    exchange = make_exchange_field(
        honeycomb,
        J_1=NiPS3_params["J1"],
        J_2=NiPS3_params["J2"],
        J_3=NiPS3_params["J3"],
    )
    np.testing.assert_allclose(exchange, exchange.T)

    simulation = LLGS_Simulation_2D(
        honeycomb,
        H_E=exchange,
        H_ext=np.array([5.0, 5.0, 0.0]),
        alpha=0.1,
        H_para=NiPS3_params["H_para"],
        H_perp=NiPS3_params["H_perp"],
        io_foldername=tmp_path,
        io_filename="custom_result",
        io_screen=False,
        method="RK4",
    )

    dt = 2e-4
    record = simulation.evolve(dt=dt, max_iters=1000)
    result_path = tmp_path / "custom_result.h5"

    assert record.shape == (1000, honeycomb.N, 3)
    assert np.isfinite(record).all()
    np.testing.assert_allclose(honeycomb.spins, initial_spins)
    assert result_path.exists()

    with h5py.File(result_path) as output:
        np.testing.assert_allclose(output["structure"][()], honeycomb.structure)
        assert output.attrs["dt"] == dt

    result = ReadResult(result_path)
    assert result.spin_datas.shape == record.shape
    np.testing.assert_allclose(result.spin_datas, record)
    np.testing.assert_allclose(result.times, np.arange(1000) * dt)
