import numpy as np
from pathlib import Path
from typing import Literal, Optional, Sequence, Tuple, Union

from numba import njit
import h5py
from scipy import sparse
from tqdm import tqdm

from .lattice import Lattice_2D
from .spin_velocity import (
    calculate_spin_velocities_csr_jit,
    calculate_spin_velocities_jit,
)

Matrix = Union[np.ndarray, sparse.spmatrix]
DMIField = Union[np.ndarray, Sequence[Matrix]]

@njit
def normalize(spins):
    return spins/np.sqrt(np.sum(spins**2, axis=1)[:,None])
    
class LLGS_Simulation_2D:

    def __init__(
        self,
        lattice: Lattice_2D,
        H_E: Optional[Matrix] = None,
        H_DMI: Optional[DMIField] = None,
        H_ext: Optional[Sequence[float]] = None,
        H_FL: Optional[Sequence[float]] = None,
        H_DL: Optional[Sequence[float]] = None,
        H_perp: float = 0,
        H_para: float = 0,
        phi_a: float = 0,
        alpha: float = 0,
        method: str = "RK4",
        io_foldername: Union[str, Path] = "lattice",
        io_filename: Optional[str] = None,
        io_compress: bool = True,
        io_screen: bool = True,
        matrix_type: Literal["sparse", "dense", "auto"] = "auto",
    ) -> None:
        self.lattice = lattice
        self.N = lattice.N
        if H_E is None:
            self.H_E = np.zeros((self.N, self.N))
        elif sparse.issparse(H_E):
            self.H_E = H_E
        else:
            self.H_E = np.asarray(H_E)
        if self.H_E.shape != (self.N, self.N):
            raise ValueError(
                f"H_E must have shape ({self.N}, {self.N}), got {self.H_E.shape}"
            )
        self.H_DMI = self._validate_dmi(H_DMI)
        if matrix_type not in ("sparse", "dense", "auto"):
            raise ValueError("matrix_type must be 'sparse', 'dense', or 'auto'")
        self.matrix_type = matrix_type
        if matrix_type == "sparse":
            self.H_E = sparse.csr_matrix(self.H_E)
            self.H_DMI = tuple(
                sparse.csr_matrix(component) for component in self.H_DMI
            )
        elif matrix_type == "dense":
            self.H_E = (
                self.H_E.toarray()
                if sparse.issparse(self.H_E)
                else np.asarray(self.H_E)
            )
            self.H_DMI = np.stack(
                [
                    component.toarray()
                    if sparse.issparse(component)
                    else np.asarray(component)
                    for component in self.H_DMI
                ]
            )
        self.H_ext = np.zeros(3) if H_ext is None else np.asarray(H_ext)
        self.H_FL = np.zeros(3) if H_FL is None else np.asarray(H_FL)
        self.H_DL = np.zeros(3) if H_DL is None else np.asarray(H_DL)
        self.H_perp = H_perp
        self.H_para = H_para
        self.phi_a = phi_a
        self.alpha = alpha
        self.method = method
        self.io_foldername = io_foldername
        self.io_filename = f"results_{method}" if io_filename is None else io_filename
        self.io_compress = io_compress
        self.io_screen = io_screen

    def _validate_dmi(self, H_DMI: Optional[DMIField]) -> DMIField:
        if H_DMI is None:
            return np.zeros((3, self.N, self.N))

        if sparse.issparse(H_DMI):
            raise ValueError("sparse H_DMI must be a sequence of three matrices")

        if isinstance(H_DMI, (list, tuple)) and any(
            sparse.issparse(component) for component in H_DMI
        ):
            if len(H_DMI) != 3:
                raise ValueError("sparse H_DMI must contain three matrices")
            for component in H_DMI:
                if np.shape(component) != (self.N, self.N):
                    raise ValueError(
                        f"each H_DMI matrix must have shape ({self.N}, {self.N})"
                    )
            return tuple(H_DMI)

        H_DMI = np.asarray(H_DMI)
        if H_DMI.shape != (3, self.N, self.N):
            raise ValueError(
                f"H_DMI must have shape (3, {self.N}, {self.N}), got {H_DMI.shape}"
            )
        return H_DMI

    def _prepare_fields(self) -> Tuple[bool, Matrix, Optional[Matrix]]:
        dmi_is_sparse = isinstance(self.H_DMI, tuple)
        use_sparse = sparse.issparse(self.H_E) or dmi_is_sparse

        if not use_sparse:
            H_E = np.asarray(self.H_E)
            H_E = (H_E + H_E.T) / 2
            H_DMI = np.asarray(self.H_DMI)
            H_DMI = (H_DMI - H_DMI.transpose(0, 2, 1)) / 2
            if np.all(H_DMI == 0):
                H_DMI = None
            return False, H_E, H_DMI

        H_E = sparse.csr_matrix(self.H_E)
        H_E = ((H_E + H_E.T) / 2).tocsr()
        H_E.eliminate_zeros()
        H_E.sort_indices()

        components = self.H_DMI if dmi_is_sparse else self.H_DMI
        normalized_dmi = []
        for component in components:
            component = sparse.csr_matrix(component)
            component = ((component - component.T) / 2).tocsr()
            component.eliminate_zeros()
            component.sort_indices()
            normalized_dmi.append(component)
        H_DMI = sparse.vstack(normalized_dmi, format="csr")
        return True, H_E, H_DMI
    
    def evolve(
        self,
        dt: float = 0.01,
        max_iters: int = 1000,
        restore_initial_state: bool = True,
    ) -> np.ndarray:
        """
        param:
        -----------------------------------------------
        dt: float
            time step per iter (ps)
        max_iters: int
            number of iteration
        restore_initial_state: bool
            If True, restore spins and velocities to their initial values after simulation

        return:
        -----------------------------------------------
        record: (max_iters, N, 3)
            1st dim: time
            2nd dim: particle idx
            3rd dim: spin (Sx,Sy,Sz)
        """
        if max_iters <= 0:
            raise ValueError("max_iters must be positive")

        if restore_initial_state:
            initial_spins = np.copy(self.lattice.spins)
            initial_svels = np.copy(self.lattice.spin_velocities)

        use_sparse, H_E, H_DMI = self._prepare_fields()

        method = self.method
        if method=="Euler":
            _get_next_spin = _get_next_spin_euler_csr if use_sparse else _get_next_spin_euler
        elif method=="RK2":
            _get_next_spin = _get_next_spin_rk2_csr if use_sparse else _get_next_spin_rk2
        elif method=="RK4":
            _get_next_spin = _get_next_spin_rk4_csr if use_sparse else _get_next_spin_rk4
        else:
            raise ValueError("method must be 'Euler','RK2','RK4'")
        
        Path(self.io_foldername).mkdir(parents=True, exist_ok=True)

        record = np.zeros((max_iters, self.N, 3))
        structure = self.lattice.structure

        iters = tqdm(range(max_iters), desc='simulation') if self.io_screen else range(max_iters)
        for n in iters:
            
            
            spins = self.lattice.spins
            svels = self.lattice.spin_velocities
            if use_sparse:
                next_spins, next_svels = _get_next_spin(
                    H_E.data, H_E.indices, H_E.indptr,
                    self.H_perp, self.H_para, self.phi_a,
                    H_DMI.data, H_DMI.indices, H_DMI.indptr, H_DMI.nnz > 0,
                    self.H_ext, self.H_FL, self.H_DL, self.alpha,
                    spins, svels, dt,
                )
            else:
                next_spins, next_svels = _get_next_spin(H_E = H_E,
                                                        H_perp = self.H_perp,
                                                        H_para = self.H_para,
                                                        phi_a= self.phi_a,
                                                        H_DMI = H_DMI,
                                                        H_ext = self.H_ext,
                                                        H_FL = self.H_FL,
                                                        H_DL = self.H_DL,
                                                        alpha = self.alpha,
                                                        spins = spins,
                                                        svels = svels,
                                                        dt = dt
                                                        )
            record[n] = spins
            self.lattice.spins = next_spins
            self.lattice.spin_velocities = next_svels
        
        if self.io_screen:
            print(f"saving data to {self.io_foldername}/{self.io_filename}.h5 ...")
            
        with h5py.File(f'{self.io_foldername}/{self.io_filename}.h5', 'w') as f:
            dataset_options = {"chunks": (min(1000, max_iters), self.N, 3)}
            if self.io_compress:
                dataset_options["compression"] = "gzip"
            f.create_dataset("spin data", data=record, **dataset_options)
            f.create_dataset("structure", structure.shape, data = structure)
            f.attrs['dt'] = dt
        
        if restore_initial_state:
            self.lattice.spins = initial_spins
            self.lattice.spin_velocities = initial_svels
        
        if self.io_screen:
            print("simulation is done")

        return record

@njit
def _get_next_spin_euler(H_E,H_perp,H_para,phi_a,H_DMI,H_ext,H_FL,H_DL,alpha,
                       spins,svels, dt):

    next_svels = calculate_spin_velocities_jit( H_E,H_perp,H_para,phi_a,H_DMI,H_ext,H_FL,H_DL,alpha,
                                                spins,svels)
    next_spins = spins + dt*svels
    next_spins = normalize(next_spins)

    return next_spins, next_svels

@njit
def _get_next_spin_rk2(H_E,H_perp,H_para,phi_a,H_DMI,H_ext,H_FL,H_DL,alpha,
                    spins,svels, dt):


    #k2
    spin2 = spins + dt/2*svels
    svel2 = calculate_spin_velocities_jit(H_E,H_perp,H_para,phi_a,H_DMI,H_ext,H_FL,H_DL,alpha,
                                        spin2,svels)
                                            
    next_spins = spins + dt/2*(svels+svel2)
    next_spins = normalize(next_spins)

    next_svels = calculate_spin_velocities_jit(H_E,H_perp,H_para,phi_a,H_DMI,H_ext,H_FL,H_DL,alpha,
                                        next_spins,svels)
    
    return next_spins, next_svels

@njit
def _get_next_spin_rk4(H_E,H_perp,H_para,phi_a,H_DMI,H_ext,H_FL,H_DL,alpha,
                    spins,svels, dt):

    spins_i = spins.copy()
    #k2
    spin2 = spins_i + dt/2*svels
    svel2 = calculate_spin_velocities_jit(H_E,H_perp,H_para,phi_a,H_DMI,H_ext,H_FL,H_DL,alpha,
                                        spin2,svels)
    
    #k3
    spin3 = spins_i + dt/2*svel2
    svel3 = calculate_spin_velocities_jit(H_E,H_perp,H_para,phi_a,H_DMI,H_ext,H_FL,H_DL,alpha,
                                        spin3,svels)
    
    #k4
    spin4 = spins_i + dt*svel3
    svel4 = calculate_spin_velocities_jit(H_E,H_perp,H_para,phi_a,H_DMI,H_ext,H_FL,H_DL,alpha,
                                        spin4,svels)
                                            


    next_spins = spins_i + dt/6*(svels+2*svel2+2*svel3+svel4)
    next_spins = normalize(next_spins)

    next_svels = calculate_spin_velocities_jit(H_E,H_perp,H_para,phi_a,H_DMI,H_ext,H_FL,H_DL,alpha,
                                            next_spins,svels)

    return next_spins, next_svels


@njit
def _get_next_spin_euler_csr(
    H_E_data, H_E_indices, H_E_indptr, H_perp, H_para, phi_a,
    H_DMI_data, H_DMI_indices, H_DMI_indptr, has_DMI,
    H_ext, H_FL, H_DL, alpha, spins, svels, dt,
):
    next_svels = calculate_spin_velocities_csr_jit(
        H_E_data, H_E_indices, H_E_indptr, H_perp, H_para, phi_a,
        H_DMI_data, H_DMI_indices, H_DMI_indptr, has_DMI,
        H_ext, H_FL, H_DL, alpha, spins, svels,
    )
    next_spins = normalize(spins + dt * svels)
    return next_spins, next_svels


@njit
def _get_next_spin_rk2_csr(
    H_E_data, H_E_indices, H_E_indptr, H_perp, H_para, phi_a,
    H_DMI_data, H_DMI_indices, H_DMI_indptr, has_DMI,
    H_ext, H_FL, H_DL, alpha, spins, svels, dt,
):
    spin2 = spins + dt / 2 * svels
    svel2 = calculate_spin_velocities_csr_jit(
        H_E_data, H_E_indices, H_E_indptr, H_perp, H_para, phi_a,
        H_DMI_data, H_DMI_indices, H_DMI_indptr, has_DMI,
        H_ext, H_FL, H_DL, alpha, spin2, svels,
    )
    next_spins = normalize(spins + dt / 2 * (svels + svel2))
    next_svels = calculate_spin_velocities_csr_jit(
        H_E_data, H_E_indices, H_E_indptr, H_perp, H_para, phi_a,
        H_DMI_data, H_DMI_indices, H_DMI_indptr, has_DMI,
        H_ext, H_FL, H_DL, alpha, next_spins, svels,
    )
    return next_spins, next_svels


@njit
def _get_next_spin_rk4_csr(
    H_E_data, H_E_indices, H_E_indptr, H_perp, H_para, phi_a,
    H_DMI_data, H_DMI_indices, H_DMI_indptr, has_DMI,
    H_ext, H_FL, H_DL, alpha, spins, svels, dt,
):
    spins_i = spins.copy()
    spin2 = spins_i + dt / 2 * svels
    svel2 = calculate_spin_velocities_csr_jit(
        H_E_data, H_E_indices, H_E_indptr, H_perp, H_para, phi_a,
        H_DMI_data, H_DMI_indices, H_DMI_indptr, has_DMI,
        H_ext, H_FL, H_DL, alpha, spin2, svels,
    )
    spin3 = spins_i + dt / 2 * svel2
    svel3 = calculate_spin_velocities_csr_jit(
        H_E_data, H_E_indices, H_E_indptr, H_perp, H_para, phi_a,
        H_DMI_data, H_DMI_indices, H_DMI_indptr, has_DMI,
        H_ext, H_FL, H_DL, alpha, spin3, svels,
    )
    spin4 = spins_i + dt * svel3
    svel4 = calculate_spin_velocities_csr_jit(
        H_E_data, H_E_indices, H_E_indptr, H_perp, H_para, phi_a,
        H_DMI_data, H_DMI_indices, H_DMI_indptr, has_DMI,
        H_ext, H_FL, H_DL, alpha, spin4, svels,
    )
    next_spins = normalize(spins_i + dt / 6 * (svels + 2 * svel2 + 2 * svel3 + svel4))
    next_svels = calculate_spin_velocities_csr_jit(
        H_E_data, H_E_indices, H_E_indptr, H_perp, H_para, phi_a,
        H_DMI_data, H_DMI_indices, H_DMI_indptr, has_DMI,
        H_ext, H_FL, H_DL, alpha, next_spins, svels,
    )
    return next_spins, next_svels
