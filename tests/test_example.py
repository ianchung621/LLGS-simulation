import h5py
import numpy as np

from llgs.LLGS_simulation import LLGS_Simulation_2D
from llgs.lattice import lattice_2D
from llgs.read_results import ReadResult
from param.NiPS3 import NiPS3_params


def make_honeycomb(n_a=3, n_b=2):
    honeycomb = lattice_2D(n_a=n_a, n_b=n_b, n_site=2)
    honeycomb.set_position(
        r_a=np.array([np.sqrt(3), 0]),
        r_b=np.array([0.5 * np.sqrt(3), 1.5]),
        r_site=np.array(
            [
                [0.5 * np.sqrt(3), 0.5],
                [np.sqrt(3), 1],
            ]
        ),
    )
    return honeycomb


def make_exchange_field(honeycomb, J_1=0, J_2=0, J_3=0):
    positions = honeycomb.get_positions()
    distances = np.linalg.norm(
        positions[:, None, :] - positions[None, :, :], axis=-1
    )
    exchange = np.zeros(distances.shape)
    exchange[np.isclose(distances, 1, atol=1e-3, rtol=0)] = J_1 / 2
    exchange[np.isclose(distances, np.sqrt(3), atol=1e-3, rtol=0)] = J_2 / 2
    exchange[np.isclose(distances, 2, atol=1e-3, rtol=0)] = J_3 / 2
    return exchange


def test_honeycomb_lattice_and_zigzag_initialization(tmp_path):
    honeycomb = make_honeycomb()

    assert honeycomb.N == 12
    assert honeycomb.get_tags().shape == (12, 3)
    assert honeycomb.get_positions().shape == (12, 2)
    assert honeycomb.get_structure().shape == (12, 5)
    np.testing.assert_allclose(
        honeycomb.get_positions()[:2],
        [[0.5 * np.sqrt(3), 0.5], [np.sqrt(3), 1]],
    )

    honeycomb.initialize_spin(
        {
            "b % 2 == 0": np.array([1, 0, 0]),
            "b % 2 == 1": np.array([-1, 0, 0]),
        }
    )
    expected_x = np.where(honeycomb.get_tags()[:, 1] % 2 == 0, 1, -1)
    np.testing.assert_array_equal(honeycomb.get_spins()[:, 0], expected_x)
    np.testing.assert_array_equal(honeycomb.get_spins()[:, 1:], 0)

    output_path = tmp_path / "honeycomb.csv"
    honeycomb.output_lattice_structure(output_path)
    assert output_path.read_text().splitlines()[0] == "particle idx,a,b,site,x,y"


def test_nips3_simulation_output_can_be_read(tmp_path):
    honeycomb = make_honeycomb(n_a=2, n_b=2)
    honeycomb.initialize_spin(
        {
            "b % 2 == 0": np.array([1, 0, 0]),
            "b % 2 == 1": np.array([-1, 0, 0]),
        }
    )
    initial_spins = honeycomb.get_spins().copy()

    exchange = make_exchange_field(
        honeycomb,
        J_1=NiPS3_params["J1"],
        J_2=NiPS3_params["J2"],
        J_3=NiPS3_params["J3"],
    )
    np.testing.assert_allclose(exchange, exchange.T)

    simulation = LLGS_Simulation_2D(honeycomb)
    simulation.set_exchange_field(exchange)
    simulation.set_H_ext(np.array([5.0, 5.0, 0.0]))
    simulation.setup(
        alpha=0.1,
        H_para=NiPS3_params["H_para"],
        H_perp=NiPS3_params["H_perp"],
        io_foldername=tmp_path,
        io_filename="results_RK4",
        io_screen=False,
        method="RK4",
    )

    dt = 2e-4
    record = simulation.evolve(honeycomb, dt=dt, max_iters=1000)
    result_path = tmp_path / "results_RK4.h5"

    assert record.shape == (1000, honeycomb.N, 3)
    assert np.isfinite(record).all()
    np.testing.assert_allclose(honeycomb.get_spins(), initial_spins)
    assert result_path.exists()

    with h5py.File(result_path) as output:
        np.testing.assert_allclose(output["structure"][()], honeycomb.get_structure())
        assert output.attrs["dt"] == dt

    result = ReadResult(result_path)
    assert result.spin_datas.shape == record.shape
    np.testing.assert_allclose(result.spin_datas, record)
    np.testing.assert_allclose(result.times, np.arange(1000) * dt)
