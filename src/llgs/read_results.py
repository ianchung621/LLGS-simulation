from typing import Literal, Union

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

    def animate(
        self,
        period,
        save_fn,
        fps=10,
        display=True,
        theme: Union[Literal["light", "dark"], dict] = "dark",
    ):
        """Animate this result and save it to a GIF or video file.

        Parameters
        ----------
        period : int
            Number of simulation steps between animation frames.
        save_fn : path-like
            Output GIF or video filename.
        fps : int, default 10
            Saved animation frame rate.
        display : bool, default True
            Show the animation figure with Matplotlib when true.
        theme : {"light", "dark"} or dict, default "dark"
            Built-in theme name or custom values overriding ``DARK_THEME``.

        Returns
        -------
        matplotlib.animation.FuncAnimation
            The generated animation.
        """
        from .plotting import _animate

        return _animate(
            self,
            period=period,
            save_fn=save_fn,
            fps=fps,
            display=display,
            theme=theme,
        )
