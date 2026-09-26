# LLGS Simulation

[![Tests](https://github.com/ianchung621/LLGS-simulation/actions/workflows/test.yml/badge.svg)](https://github.com/ianchung621/LLGS-simulation/actions/workflows/test.yml)
[![Coverage](https://codecov.io/gh/ianchung621/LLGS-simulation/branch/main/graph/badge.svg)](https://codecov.io/gh/ianchung621/LLGS-simulation)
![Python](https://img.shields.io/badge/python-3.9--3.13-blue)
![OS](https://img.shields.io/badge/OS-Linux%20%7C%20macOS%20%7C%20Windows-lightgrey)

This package provides tools for simulating and analyzing spin dynamics using
the Landau-Lifshitz-Gilbert-Slonczewski (LLGS) equation on two-dimensional
lattices. It supports antiferromagnetic exchange, DMI, anisotropy, external
fields, and spin-orbit torque.

## Install

LLGS Simulation requires Python 3.9 or newer:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install llgs-simulation
```

## Example Usage

Below is a step-by-step demonstration of how to use the repository to simulate spin dynamics and visualize the results.

### 1. Creating a Lattice Object

The `lattice.py` module allows you to define a spin lattice. Here's an example of creating a hexagonal lattice:

```python
import numpy as np

from llgs import Lattice_2D

honeycomb = Lattice_2D(
    n_a=10,
    n_b=8,
    n_site=2,
    r_a=[np.sqrt(3), 0],
    r_b=[0.5 * np.sqrt(3), 1.5],
    r_site=[
        [1 / 3, 1 / 3],
        [2 / 3, 2 / 3],
    ],
)
honeycomb.plot(draw_unitcell=True)
```

![Honeycomb lattice](https://raw.githubusercontent.com/ianchung621/LLGS-simulation/main/doc/lattice.png)

### 2. Initializing Spins

Initialize the spins on the lattice to a desired configuration:

```python
zigzag_config = {
    "b % 2 == 0": np.array([1, 0, 0]),
    "b % 2 == 1": np.array([-1, 0, 0]),
}
honeycomb.initialize_spin(zigzag_config)
honeycomb.plot()
```

![Initialized spins](https://raw.githubusercontent.com/ianchung621/LLGS-simulation/main/doc/spin_initialized.png)

### 3. Setting Up Parameters for Simulation

Build the exchange matrix from exact lattice bonds, then configure the
simulation. Each bond is `(source_site, target_site, cell_offset, coupling)`;
the reverse bond is added automatically.

```python
from llgs import LLGS_Simulation_2D, build_exchange

J1 = -11.2  # exchange-matrix coefficient, Tesla
exchange_bonds = [
    (0, 1, (0, 0), J1),
    (1, 0, (0, 1), J1),
    (1, 0, (1, 0), J1),
]
H_E = build_exchange(honeycomb, exchange_bonds)
H_ext = np.array([5.0, 5.0, 0.0])

sim = LLGS_Simulation_2D(
    honeycomb,
    H_E=H_E,
    H_ext=H_ext,
    alpha=0.1,
    H_para=0.086,
    H_perp=1.812,
    io_foldername="Data",
    io_filename="results_RK4",
    method="RK4",
)
```

`H_E` may be a dense `(N, N)` array or a SciPy sparse matrix. For sparse DMI,
pass `H_DMI` as a sequence of three sparse `(N, N)` matrices, one for each
Cartesian component. Dense DMI remains a `(3, N, N)` array. Use
`matrix_type="sparse"` to convert both fields to CSR matrices,
`matrix_type="dense"` to convert them to NumPy arrays, or the default
`matrix_type="auto"` to preserve the supplied representation.

### 4. Running the Simulation

Run the simulation for a specified number of steps and save the results:

```python
sim.evolve(dt=2e-4, max_iters=1000)
```

### 5. Visualizing Spin Dynamics

The `read_result.py` module reads the simulation results and creates visualizations. Here is how you can animate the spin dynamics:

```python
from llgs import ReadResult

results = ReadResult("Data/results_RK4.h5")

results.animate(period=50, save_fn="Data/movie.gif")
```

### Example Output

Here is an example of how the animation might look:

![Spin dynamics animation](https://raw.githubusercontent.com/ianchung621/LLGS-simulation/main/doc/spin_animation.gif)

### Additional Details

For a comprehensive explanation of the methods, equations, and parameters,
see the [example notebook](https://github.com/ianchung621/LLGS-simulation/blob/main/example.ipynb).

This notebook includes:
- Detailed descriptions of lattice construction.
- Explanation of the exchange field matrix and LLGS simulation.
- Mathematical derivations and parameter setups.
