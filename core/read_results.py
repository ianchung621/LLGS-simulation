import h5py
import numpy as np
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import matplotlib.colors as colors

class ReadResult:

    def __init__(self,fn):
        '''
        param:
        -----------------------------
        fn: hdf5 file name
        
        attr:
        -----------------------------
        times: (T)
        spin_datas: (T,N,3)
        '''
        with h5py.File(fn,'r') as f:
            self.spin_datas = f['spin data'][()] # (T,N,3)
            self.structure = f['structure'][()] # (N,5) a,b,site,x,y
            self.times = np.arange(self.spin_datas.shape[0])*f.attrs['dt'] # (T)
    
    def animate(self, period, save_fn, fps = 10):
        '''
        param:
        -----------------------------
        period: (int) period between frame
        fn: file name
        
        '''
        if self.structure.shape[1] < 5:
            raise ValueError("x,y coordinate of lattice inavailable in structure, use get_position before running LLGS simulation")
        plt.style.use("dark_background")
        plt.set_cmap('bwr')

        fig, ax = plt.subplots()
        line = ax.plot([],[])
        cmap = plt.get_cmap("bwr")
        norm = colors.Normalize(vmin=-1, vmax=1)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        plt.colorbar(sm,ax=ax)

        def init():
            return line

        def updateLattice(frame):
            frame = np.min((frame*period, len(self.spin_datas)))
            x, y = self.structure[:,3], self.structure[:,4]
            sx,sy,sz = self.spin_datas[frame].T

            arrowscale=0.3
            ax.clear()
            ax.scatter(x,y,c = 'yellow',s=1)
            ax.quiver(x,y,sx*arrowscale,sy*arrowscale,sz,norm=norm)
            ax.set_aspect('equal')
            ax.axis("off")
            plt.title(f"time = {(self.times[frame]):0.2f} ps")

            return line


        ani = animation.FuncAnimation(fig, updateLattice, frames=len(self.times)//period,init_func=init, blit=True)
        ani.save(save_fn,fps=fps)
        return ani

