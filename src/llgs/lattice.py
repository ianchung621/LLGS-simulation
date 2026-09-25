import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mplcolors
from matplotlib.patches import Polygon
import pandas as pd

def normalize(spins):
    return spins/np.linalg.norm(spins, axis = 1, keepdims=True)

class lattice_2D:

    def __init__(self,n_a,n_b,n_site):
        '''
        param 
        ------------------------------------------------
        n_a: number of Bragg basis in a-axis 
        n_b: number of Bragg basis in b-axis 
        n_site: number of particle in one Bragg basis
        
        attr:
        -------------------------------------------------
        N: number of particle
        _tags:  (N,3) index of each particle, columns: a,b,site
        _position: (N,2) for each particle
        _spins: (N,3) for each particle
        _spin_volocities: (N,3) for each particle
        '''
        self.n_site = n_site
        self.n_a = n_a
        self.n_b = n_b
        N = n_site*n_a*n_b
        self.N = N

        a, b, s = np.meshgrid( 
            np.arange(n_a), 
            np.arange(n_b), 
            np.arange(n_site),
            indexing='ij'
        )
        self._tags = np.stack((a.ravel(), b.ravel(), s.ravel()), axis=1).astype(int)
        self._positions=np.zeros((N,2))
        self._spins=np.zeros((N,3))
        self._spin_velocities=np.zeros((N,3))
        self.structure = pd.DataFrame(data = {'a':self._tags[:,0],
                                        'b':self._tags[:,1],
                                        'site':self._tags[:,2],}).astype(int)
        self.structure.index.name = 'particle idx'
        return
    
    #getter
    def get_tags(self):
        """
        -------
        (N,3):
        row: i th particle
        column: a, b, site
        """
        return self._tags
    def get_positions(self): 
        return self._positions
    def get_spins(self):
        return self._spins
    def get_spin_velocities(self):
        return self._spin_velocities
    def get_structure(self) -> pd.DataFrame:
        """
        Returns
        -------
        pd.DataFrame
            idx: particle idx
            col: a, b, site, x, y (x,y available if set_position)
        """
        return self.structure

    #setter
    def set_spins(self,spins):
        self._spins = spins
        return 
    def set_spin_velocities(self,vel):
        self._spin_velocities = vel
        return
    
    def output_lattice_structure(self,fn):
        """
        Write lattice data into a file named "fn"
        """
        self.structure.to_csv(fn)
        return
    
    
    def set_position(self, r_a:np.ndarray, r_b:np.ndarray, r_site:np.ndarray):
        """
        ----------------------------------------
        r_a: (2,) basis vector on a axis for x, y coordinate
        r_b: (2,) basis vector on b axis for x, y coordinate 
        r_site: (n_site,2) x,y coordinate for each site
        
        """
        if r_a.shape != (2,):
            raise ValueError(f"r_a must be a (2,) array, but got shape {r_a.shape}.")
        if r_b.shape != (2,):
            raise ValueError(f"r_b must be a (2,) array, but got shape {r_b.shape}.")
        # Check the dimensions of r_site
        if r_site.shape != (self.n_site, 2):
            raise ValueError(f"r_site must be a ({self.n_site}, 2) array, but got shape {r_site.shape}.")

        self.r_a = r_a
        self.r_b = r_b
        self.r_site = r_site


        a, b, s = np.meshgrid(
            np.arange(self.n_a), 
            np.arange(self.n_b), 
            np.arange(self.n_site), 
            indexing='ij'
        )
        
        a = a.flatten()
        b = b.flatten()
        s = s.flatten()

        if self.n_site == 1:
            
            r_site_x, r_site_y = r_site[0]  
            self._positions[:, 0] = a * r_a[0] + b * r_b[0] + r_site_x
            self._positions[:, 1] = a * r_a[1] + b * r_b[1] + r_site_y
        else:
            self._positions[:, 0] = a * r_a[0] + b * r_b[0] + r_site[s, 0]
            self._positions[:, 1] = a * r_a[1] + b * r_b[1] + r_site[s, 1]
        self.structure['x'] = self._positions[:, 0]
        self.structure['y'] = self._positions[:, 1]
        return

    def initialize_spin(self, condition_dict, perturb=0.0):
        """
        Parameters:
        ----------
        conditions : dict[str, np.ndarray (3)]
            A dictionary mapping string conditions (evaluated on `structure`) to spin vectors.
        """
        
        for cond_str, spin in condition_dict.items():
            cond = self.structure.eval(cond_str)
            self._spins[cond] = spin
        
        self._spins +=  perturb * np.random.normal(0,1,(self.N,3))
        self._spins = normalize(self._spins)

        return



    def plot(self, arrowscale=0.3, annotate_idx = False, draw_unitcell = False):
        x = self.get_positions()[:,0]
        y = self.get_positions()[:,1]
        sx = self.get_spins()[:,0]
        sy = self.get_spins()[:,1]
        sz = self.get_spins()[:,2]

        plt.style.use("dark_background")
        plt.set_cmap('bwr')
        fig, ax =plt.subplots()
        cmap = plt.get_cmap("bwr")
        norm = mplcolors.Normalize(vmin=-1, vmax=1)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        plt.colorbar(sm,ax=ax)
        ax.scatter(x,y,c = 'yellow', s = 2)
        if np.linalg.norm(sx) == np.linalg.norm(sy) == 0 and np.linalg.norm(sz) != 0:
            ax.scatter(x,y,c=sz, s = 2, norm=norm)
        elif np.linalg.norm(sx) != 0 or np.linalg.norm(sy) != 0:
            ax.quiver(x,y,sx*arrowscale,sy*arrowscale,sz,norm=norm)
        if annotate_idx:
            for (i, (x,y)) in enumerate(zip(x,y)):
                ax.annotate(str(i), (x,y),  xycoords='data',
                xytext=(1.5, 1.5), textcoords='offset points')
        
        if draw_unitcell:
            origin = np.zeros(2)
            v1 = self.r_a
            v2 = self.r_b
            v3 = v1 + v2
            unit_cell = Polygon([origin, v1, v3, v2], closed=True, edgecolor='blue', facecolor='lightblue', alpha=0.5)
            ax.add_patch(unit_cell)

        ax.set_aspect('equal')
        ax.axis("off")
        
        return













