import importlib

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pytest
from scipy import sparse

from llgs import LLGS_Simulation_2D, Lattice_2D, ReadResult
simulation_module = importlib.import_module("llgs.LLGS_simulation")

NIPS3_PARAMS = {
    "J1": -22.432766768116426,
    "J2": 1.7255974437012638,
    "J3": 117.34062617168593,
    "H_para": 0.08627987218506318,
    "H_perp": 1.8118773158863268,
}


def make_honeycomb(n_a=3, n_b=2):
    return Lattice_2D(
        n_a=n_a,
        n_b=n_b,
        n_site=2,
        r_a=np.array([np.sqrt(3), 0]),
        r_b=np.array([0.5 * np.sqrt(3), 1.5]),
        r_site=np.array(
            [
                [1 / 3, 1 / 3],
                [2 / 3, 2 / 3],
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
    lattice = Lattice_2D(n_a=2, n_b=1, n_site=1)

    assert lattice.positions.shape == (2, 2)
    assert lattice.structure.shape == (2, 3)
    assert not lattice.has_geometry

    with pytest.raises(ValueError, match="must be provided together"):
        Lattice_2D(
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
    np.testing.assert_allclose(honeycomb.r_site, [[1 / 3, 1 / 3], [2 / 3, 2 / 3]])

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

    figure, axes = honeycomb.plot(draw_unitcell=True, display=False)
    assert figure.axes[0] is axes
    assert plt.get_fignums()
    plt.close("all")


def test_fields_are_normalized_when_evolution_starts(tmp_path, monkeypatch):
    lattice = Lattice_2D(n_a=1, n_b=1, n_site=2)
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
    simulation.evolve(max_iters=1)

    np.testing.assert_array_equal(simulation.H_E, raw_exchange)
    np.testing.assert_array_equal(simulation.H_DMI, raw_dmi)
    np.testing.assert_allclose(captured["H_E"], [[0, 3], [3, 0]])
    np.testing.assert_allclose(captured["H_DMI"][0], [[0, 1], [-1, 0]])


def test_short_uncompressed_simulation_output(tmp_path):
    lattice = Lattice_2D(n_a=1, n_b=1, n_site=1)
    lattice.spins[:] = [1.0, 0.0, 0.0]
    simulation = LLGS_Simulation_2D(
        lattice,
        method="Euler",
        io_foldername=tmp_path,
        io_filename="uncompressed",
        io_compress=False,
        io_screen=False,
    )

    record = simulation.evolve(max_iters=1)

    assert record.shape == (1, 1, 3)
    with h5py.File(tmp_path / "uncompressed.h5") as output:
        assert output["spin data"].compression is None
        assert output["spin data"].chunks == (1, 1, 3)

    with pytest.raises(ValueError, match="max_iters must be positive"):
        simulation.evolve(max_iters=0)


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
        J_1=NIPS3_PARAMS["J1"],
        J_2=NIPS3_PARAMS["J2"],
        J_3=NIPS3_PARAMS["J3"],
    )
    np.testing.assert_allclose(exchange, exchange.T)

    simulation = LLGS_Simulation_2D(
        honeycomb,
        H_E=exchange,
        H_ext=np.array([5.0, 5.0, 0.0]),
        alpha=0.1,
        H_para=NIPS3_PARAMS["H_para"],
        H_perp=NIPS3_PARAMS["H_perp"],
        io_foldername=tmp_path,
        io_filename="custom_result",
        io_screen=False,
        method="RK4",
    )

    dt = 2e-4
    max_iters = 10
    record = simulation.evolve(dt=dt, max_iters=max_iters)
    result_path = tmp_path / "custom_result.h5"

    assert record.shape == (max_iters, honeycomb.N, 3)
    assert np.isfinite(record).all()
    np.testing.assert_allclose(honeycomb.spins, initial_spins)
    assert result_path.exists()

    with h5py.File(result_path) as output:
        np.testing.assert_allclose(output["structure"][()], honeycomb.structure)
        assert output["spin data"].compression == "gzip"
        assert output["spin data"].chunks == (max_iters, honeycomb.N, 3)
        assert output.attrs["dt"] == dt

    result = ReadResult(result_path)
    assert result.spin_datas.shape == record.shape
    np.testing.assert_allclose(result.spin_datas, record)
    np.testing.assert_allclose(result.times, np.arange(max_iters) * dt)


@pytest.mark.parametrize("sparse_exchange", [False, True])
@pytest.mark.parametrize("sparse_dmi", [False, True])
def test_sparse_and_dense_fields_produce_same_evolution(
    tmp_path, sparse_exchange, sparse_dmi
):
    dense_exchange = np.array([[0.0, 0.4], [0.2, 0.0]])
    dense_dmi = np.array(
        [
            [[0.0, 0.05], [-0.01, 0.0]],
            [[0.0, -0.03], [0.01, 0.0]],
            [[0.0, 0.02], [-0.04, 0.0]],
        ]
    )

    def run(H_E, H_DMI, filename):
        lattice = Lattice_2D(n_a=1, n_b=1, n_site=2)
        lattice.spins[:] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        simulation = LLGS_Simulation_2D(
            lattice,
            H_E=H_E,
            H_DMI=H_DMI,
            H_ext=np.array([0.1, 0.0, 0.2]),
            alpha=0.1,
            method="RK4",
            io_foldername=tmp_path,
            io_filename=filename,
            io_screen=False,
        )
        return simulation.evolve(dt=1e-3, max_iters=5)

    expected = run(dense_exchange, dense_dmi, "dense")
    H_E = sparse.csr_matrix(dense_exchange) if sparse_exchange else dense_exchange
    H_DMI = (
        tuple(sparse.csr_matrix(component) for component in dense_dmi)
        if sparse_dmi
        else dense_dmi
    )
    actual = run(H_E, H_DMI, f"sparse-{sparse_exchange}-{sparse_dmi}")
    np.testing.assert_allclose(actual, expected)


def test_sparse_field_validation():
    lattice = Lattice_2D(n_a=1, n_b=1, n_site=2)
    with pytest.raises(ValueError, match="H_E must have shape"):
        LLGS_Simulation_2D(lattice, H_E=sparse.eye(3))
    with pytest.raises(ValueError, match="sequence of three"):
        LLGS_Simulation_2D(lattice, H_DMI=sparse.eye(2))
    with pytest.raises(ValueError, match="contain three"):
        LLGS_Simulation_2D(lattice, H_DMI=(sparse.eye(2), sparse.eye(2)))


def test_sparse_fields_are_normalized_without_densifying():
    lattice = Lattice_2D(n_a=1, n_b=1, n_site=2)
    exchange = sparse.csr_matrix([[0.0, 4.0], [2.0, 0.0]])
    dmi = (
        sparse.csr_matrix([[0.0, 4.0], [2.0, 0.0]]),
        sparse.csr_matrix((2, 2)),
        sparse.csr_matrix((2, 2)),
    )
    simulation = LLGS_Simulation_2D(lattice, H_E=exchange, H_DMI=dmi)

    use_sparse, normalized_exchange, normalized_dmi = simulation._prepare_fields()

    assert use_sparse
    assert sparse.isspmatrix_csr(normalized_exchange)
    assert sparse.isspmatrix_csr(normalized_dmi)
    np.testing.assert_allclose(normalized_exchange.toarray(), [[0, 3], [3, 0]])
    np.testing.assert_allclose(
        normalized_dmi[:2].toarray(), [[0, 1], [-1, 0]]
    )


def test_sparse_matrix_type_converts_dense_fields_to_csr():
    lattice = Lattice_2D(n_a=1, n_b=1, n_site=2)
    exchange = np.array([[0.0, 1.0], [1.0, 0.0]])
    dmi = np.zeros((3, 2, 2))
    dmi[2] = [[0.0, 0.2], [-0.2, 0.0]]

    simulation = LLGS_Simulation_2D(
        lattice,
        H_E=exchange,
        H_DMI=dmi,
        matrix_type="sparse",
    )

    assert simulation.matrix_type == "sparse"
    assert sparse.isspmatrix_csr(simulation.H_E)
    assert isinstance(simulation.H_DMI, tuple)
    assert all(sparse.isspmatrix_csr(component) for component in simulation.H_DMI)
    np.testing.assert_allclose(simulation.H_E.toarray(), exchange)
    for actual, expected in zip(simulation.H_DMI, dmi):
        np.testing.assert_allclose(actual.toarray(), expected)


def test_dense_matrix_type_converts_sparse_fields_to_arrays():
    lattice = Lattice_2D(n_a=1, n_b=1, n_site=2)
    exchange = sparse.csr_matrix([[0.0, 1.0], [1.0, 0.0]])
    dmi = tuple(sparse.csr_matrix((2, 2)) for _ in range(3))

    simulation = LLGS_Simulation_2D(
        lattice,
        H_E=exchange,
        H_DMI=dmi,
        matrix_type="dense",
    )

    assert simulation.matrix_type == "dense"
    assert isinstance(simulation.H_E, np.ndarray)
    assert isinstance(simulation.H_DMI, np.ndarray)
    assert simulation.H_DMI.shape == (3, 2, 2)


def test_matrix_type_validation():
    lattice = Lattice_2D(n_a=1, n_b=1, n_site=1)
    with pytest.raises(ValueError, match="matrix_type"):
        LLGS_Simulation_2D(lattice, matrix_type="invalid")
