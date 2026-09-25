import numpy as np
from numba import njit


@njit
def calculate_spin_velocities_jit(H_E,H_perp,H_para,phi_a,H_DMI,H_ext,H_FL,H_DL,alpha
                                  ,spins,svels):
    # spins: (N,3) svels: (N,3)
    gyro_magnetic_ratio = - 1.76085963023e-1
    H_eff = H_eff_no_DMI(H_E,H_perp,H_para,phi_a,H_ext,H_FL,spins)

    if not H_DMI is None:
        H_eff += H_eff_DMI_term(H_DMI, spins)
    
    next_svels = gyro_magnetic_ratio * (
                    np.cross(H_eff, spins)
                    + np.cross(np.cross(H_DL, spins), spins)
                    + alpha * np.cross(spins, svels)
                )

    return next_svels

@njit
def H_eff_no_DMI(H_E,H_perp,H_para,phi_a,H_ext,H_FL,spins):
    H_eff = ( H_ext # applied field
            + H_FL # Field-like SOT
            + spins * np.array([2*H_para*np.cos(phi_a), 2*H_para*np.sin(phi_a), -2*H_perp]) # anisotropy
            - H_E @ spins) # exchange 
    return H_eff

@njit
def H_eff_DMI_term(H_DMI, spins):
    epsilon = np.zeros((3, 3, 3))
    epsilon[0, 1, 2] = epsilon[1, 2, 0] = epsilon[2, 0, 1] = 1
    epsilon[0, 2, 1] = epsilon[1, 0, 2] = epsilon[2, 1, 0] = -1
    return (- H_DMI[0] @ spins @ epsilon[0]
            - H_DMI[1] @ spins @ epsilon[1]
            - H_DMI[2] @ spins @ epsilon[2]
            )


@njit
def csr_matmul_spins(data, indices, indptr, spins):
    """Multiply a CSR matrix by an (N, 3) spin array."""
    result = np.zeros((indptr.size - 1, 3), dtype=spins.dtype)
    for row in range(indptr.size - 1):
        for index in range(indptr[row], indptr[row + 1]):
            result[row] += data[index] * spins[indices[index]]
    return result


@njit
def calculate_spin_velocities_csr_jit(
    H_E_data,
    H_E_indices,
    H_E_indptr,
    H_perp,
    H_para,
    phi_a,
    H_DMI_data,
    H_DMI_indices,
    H_DMI_indptr,
    has_DMI,
    H_ext,
    H_FL,
    H_DL,
    alpha,
    spins,
    svels,
):
    """CSR equivalent of calculate_spin_velocities_jit."""
    gyro_magnetic_ratio = -1.76085963023e-1
    H_eff = (
        H_ext
        + H_FL
        + spins
        * np.array(
            [2 * H_para * np.cos(phi_a), 2 * H_para * np.sin(phi_a), -2 * H_perp]
        )
        - csr_matmul_spins(H_E_data, H_E_indices, H_E_indptr, spins)
    )

    if has_DMI:
        dmi_products = csr_matmul_spins(
            H_DMI_data, H_DMI_indices, H_DMI_indptr, spins
        )
        n_spins = spins.shape[0]
        epsilon = np.zeros((3, 3, 3))
        epsilon[0, 1, 2] = epsilon[1, 2, 0] = epsilon[2, 0, 1] = 1
        epsilon[0, 2, 1] = epsilon[1, 0, 2] = epsilon[2, 1, 0] = -1
        for component in range(3):
            start = component * n_spins
            H_eff -= dmi_products[start : start + n_spins] @ epsilon[component]

    return gyro_magnetic_ratio * (
        np.cross(H_eff, spins)
        + np.cross(np.cross(H_DL, spins), spins)
        + alpha * np.cross(spins, svels)
    )
