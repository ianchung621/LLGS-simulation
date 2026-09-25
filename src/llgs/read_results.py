import h5py
import numpy as np

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

    def animate(self, period, save_fn, fps=10):
        """Animate this result and save it to a GIF or video file."""
        from .plotting import _animate

        return _animate(self, period=period, save_fn=save_fn, fps=fps)
