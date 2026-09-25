import numpy as np
from numba import njit
from scipy import sparse

from llgs.LLGS_simulation import calculate_spin_velocities_jit
from llgs.spin_velocity import calculate_spin_velocities_csr_jit, csr_matmul_spins


@njit
def calculate_spin_velocities_jit_GT(
    H_E,
    H_perp,
    H_para,
    phi_a,
    H_DMI,
    H_ext,
    H_FL,
    H_DL,
    alpha,
    spins,
    svels,
):
    gyro_magnetic_ratio = -1.76085963023e-1
    if H_DMI is None:
        H_eff = (
            H_ext
            + H_FL
            + spins
            * np.array(
                [2 * H_para * np.cos(phi_a), 2 * H_para * np.sin(phi_a), -2 * H_perp]
            )
            - H_E @ spins
        )
    else:
        epsilon = np.zeros((3, 3, 3))
        epsilon[0, 1, 2] = epsilon[1, 2, 0] = epsilon[2, 0, 1] = 1
        epsilon[0, 2, 1] = epsilon[1, 0, 2] = epsilon[2, 1, 0] = -1

        H_eff = (
            H_ext
            + H_FL
            + spins
            * np.array(
                [2 * H_para * np.cos(phi_a), 2 * H_para * np.sin(phi_a), -2 * H_perp]
            )
            - H_E @ spins
            - H_DMI[0] @ spins @ epsilon[0]
            - H_DMI[1] @ spins @ epsilon[1]
            - H_DMI[2] @ spins @ epsilon[2]
        )

    return gyro_magnetic_ratio * (
        np.cross(H_eff, spins)
        + np.cross(np.cross(H_DL, spins), spins)
        + alpha * np.cross(spins, svels)
    )


def test_calculate_spin_velocities_with_small_arrays():
    inputs = {
        "H_E": np.array([[0.0, 0.3], [0.3, 0.0]]),
        "H_perp": 0.2,
        "H_para": 0.1,
        "phi_a": 0.3,
        "H_ext": np.array([0.1, -0.2, 0.3]),
        "H_FL": np.array([-0.05, 0.02, 0.01]),
        "H_DL": np.array([0.03, -0.01, 0.02]),
        "alpha": 0.05,
        "spins": np.array([[1.0, 0.0, 0.0], [0.0, 0.6, 0.8]]),
        "svels": np.array([[0.01, 0.02, -0.03], [-0.02, 0.01, 0.015]]),
    }
    dmi = np.array(
        [
            [[0.0, 0.05], [-0.05, 0.0]],
            [[0.0, -0.02], [0.02, 0.0]],
            [[0.0, 0.04], [-0.04, 0.0]],
        ]
    )

    for H_DMI in (None, dmi):
        expected = calculate_spin_velocities_jit_GT(H_DMI=H_DMI, **inputs)
        actual = calculate_spin_velocities_jit(H_DMI=H_DMI, **inputs)
        np.testing.assert_allclose(actual, expected)


def test_csr_matmul_spins_matches_scipy():
    matrix = sparse.csr_matrix([[0.0, 2.0], [-1.0, 0.0], [3.0, 4.0]])
    spins = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    actual = csr_matmul_spins(matrix.data, matrix.indices, matrix.indptr, spins)
    np.testing.assert_allclose(actual, matrix @ spins)


def test_sparse_spin_velocities_match_dense():
    H_E = np.array([[0.0, 0.3], [0.3, 0.0]])
    H_DMI = np.array(
        [
            [[0.0, 0.05], [-0.05, 0.0]],
            [[0.0, -0.02], [0.02, 0.0]],
            [[0.0, 0.04], [-0.04, 0.0]],
        ]
    )
    dmi_csr = sparse.vstack([sparse.csr_matrix(x) for x in H_DMI], format="csr")
    exchange_csr = sparse.csr_matrix(H_E)
    common = {
        "H_perp": 0.2,
        "H_para": 0.1,
        "phi_a": 0.3,
        "H_ext": np.array([0.1, -0.2, 0.3]),
        "H_FL": np.array([-0.05, 0.02, 0.01]),
        "H_DL": np.array([0.03, -0.01, 0.02]),
        "alpha": 0.05,
        "spins": np.array([[1.0, 0.0, 0.0], [0.0, 0.6, 0.8]]),
        "svels": np.array([[0.01, 0.02, -0.03], [-0.02, 0.01, 0.015]]),
    }
    expected = calculate_spin_velocities_jit(H_E=H_E, H_DMI=H_DMI, **common)
    actual = calculate_spin_velocities_csr_jit(
        exchange_csr.data,
        exchange_csr.indices,
        exchange_csr.indptr,
        common["H_perp"],
        common["H_para"],
        common["phi_a"],
        dmi_csr.data,
        dmi_csr.indices,
        dmi_csr.indptr,
        True,
        common["H_ext"],
        common["H_FL"],
        common["H_DL"],
        common["alpha"],
        common["spins"],
        common["svels"],
    )
    np.testing.assert_allclose(actual, expected)
