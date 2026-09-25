import numpy as np
from pathlib import Path
from numba import njit
from .lattice import lattice_2D
import h5py
from tqdm import tqdm


@njit
def normalize(spins):
    return spins/np.sqrt(np.sum(spins**2, axis=1)[:,None])

@njit
def calculate_spin_velocities_jit(H_E,H_perp,H_para,phi_a,H_DMI,H_ext,H_FL,H_DL,alpha
                                  ,spins,svels):
    # spins: (N,3) svels: (N,3)
    gyro_magnetic_ratio = - 1.76085963023e-1
    if H_DMI is None:
        H_eff = ( H_ext # applied field
                + H_FL # Field-like SOT
                + spins * np.array([2*H_para*np.cos(phi_a), 2*H_para*np.sin(phi_a), -2*H_perp]) # anisotropy
                - H_E @ spins) # exchange 
    else:
        epsilon = np.zeros((3, 3, 3))
        epsilon[0, 1, 2] = epsilon[1, 2, 0] = epsilon[2, 0, 1] = 1
        epsilon[0, 2, 1] = epsilon[1, 0, 2] = epsilon[2, 1, 0] = -1

        H_eff = ( H_ext # applied field
                + H_FL # Field-like SOT
                + spins * np.array([2*H_para*np.cos(phi_a), 2*H_para*np.sin(phi_a), -2*H_perp]) # anisotropy
                - H_E @ spins # exchange 
                - H_DMI[0] @ spins @ epsilon[0]
                - H_DMI[1] @ spins @ epsilon[1]
                - H_DMI[2] @ spins @ epsilon[2]
                )
    
    next_svels = (gyro_magnetic_ratio*(np.cross(H_eff, spins) + np.cross(np.cross(H_DL, spins), spins)
                + alpha*np.cross(spins, svels)))

    return next_svels
    
class LLGS_Simulation_2D:

    def __init__(
        self,
        lattice: lattice_2D,
        *,
        H_E=None,
        H_DMI_x=None,
        H_DMI_y=None,
        H_DMI_z=None,
        H_ext=None,
        H_FL=None,
        H_DL=None,
        H_perp=0,
        H_para=0,
        phi_a=0,
        alpha=0,
        method="RK4",
        io_foldername="lattice",
        io_filename=None,
        io_compress=True,
        io_screen=True,
    ):
        self.lattice = lattice
        self.N = lattice.N
        self.H_E = np.zeros((self.N, self.N)) if H_E is None else np.asarray(H_E)
        self.H_DMI_x = (
            np.zeros((self.N, self.N)) if H_DMI_x is None else np.asarray(H_DMI_x)
        )
        self.H_DMI_y = (
            np.zeros((self.N, self.N)) if H_DMI_y is None else np.asarray(H_DMI_y)
        )
        self.H_DMI_z = (
            np.zeros((self.N, self.N)) if H_DMI_z is None else np.asarray(H_DMI_z)
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
    
    def evolve(self, dt:float=0.01, max_iters:int = 1000,
               restore_initial_state = True):
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
        if restore_initial_state:
            initial_spins = np.copy(self.lattice.spins)
            initial_svels = np.copy(self.lattice.spin_velocities)

        method = self.method
        if method=="Euler":
            _get_next_spin = _get_next_spin_euler
        elif method=="RK2":
            _get_next_spin = _get_next_spin_rk2
        elif method=="RK4":
            _get_next_spin = _get_next_spin_rk4    
        else:
            raise ValueError("method must be 'Euler','RK2','RK4'")
        
        Path(self.io_foldername).mkdir(parents=True, exist_ok=True)

        record = np.zeros((max_iters, self.N, 3))
        structure = self.lattice.structure

        H_E = np.asarray(self.H_E)
        H_E = (H_E + H_E.T) / 2

        H_DMI_components = []
        for component in (self.H_DMI_x, self.H_DMI_y, self.H_DMI_z):
            component = np.asarray(component)
            H_DMI_components.append((component - component.T) / 2)

        if all(np.all(component == 0) for component in H_DMI_components):
            H_DMI = None
        else:
            H_DMI = np.stack(H_DMI_components)
        
        iters = tqdm(range(max_iters), desc='simulation') if self.io_screen else range(max_iters)
        for n in iters:
            
            
            spins = self.lattice.spins
            svels = self.lattice.spin_velocities
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
            if self.io_compress:
                f.create_dataset("spin data", (max_iters, self.N, 3), data = record, 
                                chunks = (1000, self.N, 3))
            else:
                f.create_dataset("spin data", (max_iters, self.N, 3), data = record, 
                                chunks = (1000, self.N, 3))
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

    spins_i = spins
    #k2
    spins += dt/2*svels
    svel2 = calculate_spin_velocities_jit(H_E,H_perp,H_para,phi_a,H_DMI,H_ext,H_FL,H_DL,alpha,
                                        spins,svels)
    
    #k3
    spins += dt/2*svel2
    svel3 = calculate_spin_velocities_jit(H_E,H_perp,H_para,phi_a,H_DMI,H_ext,H_FL,H_DL,alpha,
                                        spins,svels)                                            
    
    #k4
    spins += dt*svels
    svel4 = calculate_spin_velocities_jit(H_E,H_perp,H_para,phi_a,H_DMI,H_ext,H_FL,H_DL,alpha,
                                        spins,svels)
                                            


    next_spins = spins_i + dt/6*(svels+2*svel2+2*svel3+svel4)
    next_spins = normalize(next_spins)

    next_svels = calculate_spin_velocities_jit(H_E,H_perp,H_para,phi_a,H_DMI,H_ext,H_FL,H_DL,alpha,
                                            next_spins,svels)

    return next_spins, next_svels
