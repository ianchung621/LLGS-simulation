import numpy as np
import pytest
from scipy import sparse

from llgs import Lattice_2D, build_dmi, build_exchange


def test_exchange_bonds_are_repeated_and_symmetric():
    lattice = Lattice_2D(n_a=2, n_b=1, n_site=2)
    exchange = build_exchange(
        lattice,
        [
            (0, 1, (0, 0), 2.0),
            (1, 0, (1, 0), 3.0),
        ],
    )
    expected = np.array(
        [
            [0, 2, 0, 0],
            [2, 0, 3, 0],
            [0, 3, 0, 2],
            [0, 0, 2, 0],
        ]
    )

    assert sparse.isspmatrix_csr(exchange)
    np.testing.assert_allclose(exchange.toarray(), expected)


def test_periodic_exchange_wraps_selected_axes():
    lattice = Lattice_2D(n_a=2, n_b=1, n_site=2)
    exchange = build_exchange(
        lattice,
        [(1, 0, (1, 0), 3.0)],
        periodic=(True, False),
    )

    assert exchange[1, 2] == 3.0
    assert exchange[3, 0] == 3.0
    np.testing.assert_allclose(exchange.toarray(), exchange.toarray().T)


def test_dmi_bonds_are_antisymmetric():
    lattice = Lattice_2D(n_a=1, n_b=1, n_site=2)
    dmi = build_dmi(lattice, [(0, 1, (0, 0), (1.0, -2.0, 3.0))])

    assert len(dmi) == 3
    assert all(sparse.isspmatrix_csr(component) for component in dmi)
    for component, value in zip(dmi, (1.0, -2.0, 3.0)):
        np.testing.assert_allclose(component.toarray(), [[0, value], [-value, 0]])


def test_interaction_bond_validation():
    lattice = Lattice_2D(n_a=1, n_b=1, n_site=2)
    with pytest.raises(ValueError, match="outside the unit cell"):
        build_exchange(lattice, [(0, 2, (0, 0), 1.0)])
    with pytest.raises(ValueError, match="cannot connect a site to itself"):
        build_exchange(lattice, [(0, 0, (0, 0), 1.0)])
    with pytest.raises(ValueError, match=r"shape \(3,\)"):
        build_dmi(lattice, [(0, 1, (0, 0), (1.0, 2.0))])
