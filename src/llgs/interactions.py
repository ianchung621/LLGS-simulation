"""Sparse interaction-matrix builders for periodic lattice topologies."""

from typing import Iterable, Sequence, Tuple, Union

import numpy as np
from scipy import sparse

from .lattice import Lattice_2D


CellOffset = Tuple[int, int]
ExchangeBond = Tuple[int, int, CellOffset, float]
DMIBond = Tuple[int, int, CellOffset, Sequence[float]]
Periodic = Union[bool, Tuple[bool, bool]]


def _periodic_axes(periodic: Periodic) -> Tuple[bool, bool]:
    if isinstance(periodic, (bool, np.bool_)):
        return bool(periodic), bool(periodic)
    if len(periodic) != 2:
        raise ValueError("periodic must be a bool or a pair of bools")
    return bool(periodic[0]), bool(periodic[1])


def _validate_bond(
    lattice: Lattice_2D,
    source_site: int,
    target_site: int,
    offset: CellOffset,
) -> Tuple[int, int]:
    if not 0 <= source_site < lattice.n_site:
        raise ValueError(f"source site {source_site} is outside the unit cell")
    if not 0 <= target_site < lattice.n_site:
        raise ValueError(f"target site {target_site} is outside the unit cell")
    if len(offset) != 2 or any(int(value) != value for value in offset):
        raise ValueError("bond cell offsets must contain two integers")
    if source_site == target_site and tuple(offset) == (0, 0):
        raise ValueError("a bond cannot connect a site to itself in the same cell")
    return int(offset[0]), int(offset[1])


def _particle_index(lattice: Lattice_2D, a: int, b: int, site: int) -> int:
    return (a * lattice.n_b + b) * lattice.n_site + site


def _bond_indices(
    lattice: Lattice_2D,
    source_site: int,
    target_site: int,
    offset: CellOffset,
    periodic: Tuple[bool, bool],
):
    delta_a, delta_b = _validate_bond(
        lattice, source_site, target_site, offset
    )
    for a in range(lattice.n_a):
        for b in range(lattice.n_b):
            target_a = a + delta_a
            target_b = b + delta_b

            if periodic[0]:
                target_a %= lattice.n_a
            elif not 0 <= target_a < lattice.n_a:
                continue

            if periodic[1]:
                target_b %= lattice.n_b
            elif not 0 <= target_b < lattice.n_b:
                continue

            yield (
                _particle_index(lattice, a, b, source_site),
                _particle_index(lattice, target_a, target_b, target_site),
            )


def build_exchange(
    lattice: Lattice_2D,
    bonds: Iterable[ExchangeBond],
    periodic: Periodic = False,
) -> sparse.csr_matrix:
    """Build a symmetric CSR exchange matrix from unit-cell bond templates.

    Each bond is ``(source_site, target_site, (delta_a, delta_b), coupling)``.
    A template is repeated from every unit cell and its reverse matrix entry is
    added automatically. With open boundaries, bonds leaving the lattice are
    omitted. ``periodic`` may be one bool or ``(periodic_a, periodic_b)``.
    """
    periodic_axes = _periodic_axes(periodic)
    rows = []
    columns = []
    values = []

    for source_site, target_site, offset, coupling in bonds:
        for source, target in _bond_indices(
            lattice, source_site, target_site, offset, periodic_axes
        ):
            rows.extend((source, target))
            columns.extend((target, source))
            values.extend((coupling, coupling))

    matrix = sparse.coo_matrix(
        (values, (rows, columns)), shape=(lattice.N, lattice.N), dtype=float
    ).tocsr()
    matrix.eliminate_zeros()
    return matrix


def build_dmi(
    lattice: Lattice_2D,
    bonds: Iterable[DMIBond],
    periodic: Periodic = False,
) -> Tuple[sparse.csr_matrix, sparse.csr_matrix, sparse.csr_matrix]:
    """Build three antisymmetric CSR DMI matrices from bond templates.

    Each bond is ``(source_site, target_site, (delta_a, delta_b), (Dx, Dy,
    Dz))``. The reverse bond is assigned the negated DMI vector. Boundary
    behavior matches :func:`build_exchange`.
    """
    periodic_axes = _periodic_axes(periodic)
    rows = []
    columns = []
    component_values = ([], [], [])

    for source_site, target_site, offset, dmi_vector in bonds:
        dmi_vector = np.asarray(dmi_vector, dtype=float)
        if dmi_vector.shape != (3,):
            raise ValueError("each DMI vector must have shape (3,)")
        for source, target in _bond_indices(
            lattice, source_site, target_site, offset, periodic_axes
        ):
            rows.extend((source, target))
            columns.extend((target, source))
            for component, values in enumerate(component_values):
                values.extend((dmi_vector[component], -dmi_vector[component]))

    matrices = []
    for values in component_values:
        matrix = sparse.coo_matrix(
            (values, (rows, columns)), shape=(lattice.N, lattice.N), dtype=float
        ).tocsr()
        matrix.eliminate_zeros()
        matrices.append(matrix)
    return tuple(matrices)
