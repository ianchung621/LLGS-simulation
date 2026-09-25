import numpy as np
from numba import njit


@njit
def calculate_spin_velocities_jit(H_E,H_perp,H_para,phi_a,H_DMI,H_ext,H_FL,H_DL,alpha
                                  ,spins,svels):
    # spins: (N,3) svels: (N,3)
    gyro_magnetic_ratio = - 1.76085963023e-1
    if H_DMI is None:
        H_eff = H_eff_no_DMI(H_E,H_perp,H_para,phi_a,H_ext,H_FL,spins)
    else:
        H_eff = (H_eff_no_DMI(H_E,H_perp,H_para,phi_a,H_ext,H_FL,spins)
                +H_eff_DMI_term(H_DMI, spins)
                )
    
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