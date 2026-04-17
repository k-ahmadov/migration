# %%
from pathlib import Path
from typing import cast

import h5py
import matplotlib.pyplot as plt
import numpy as np

# %%
result_dir = Path.cwd() / "results" / "3dec" / "runs"
fp = result_dir / "run-q-1e-06.hdf5"
with h5py.File(fp, "r") as f:
    x_sc = cast(h5py.Dataset, f["coordinates/x_subcontacts"])[:]
    x_vert = cast(h5py.Dataset, f["coordinates/x_vertices"])[:]
    t = cast(h5py.Dataset, f["coordinates/t"])[:]
    sn_tx = cast(h5py.Dataset, f["fields/stress_normal"])[:]
    p_tx = cast(h5py.Dataset, f["fields/fluid_pressure"])[:]
    w_tx = cast(h5py.Dataset, f["fields/aperture"])[:]
    params = {
        k: cast(h5py.Dataset, f[f"parameters/{k}"])[()]
        for k in cast(h5py.Group, f["parameters"]).keys()
    }

# %%

# sort the distance array
x_vert_sorted = np.sort(x_vert)
# sort each row (timestep) of w_tx according to the sorted distance
w_tx_argsorted = w_tx[:, np.argsort(x_vert)]
# value of aperture to track as the front
w_front = params["w_i"] * (1 + 0.01)
# boolean array - if aperture array is less than the chosen w_front
less_than = w_tx_argsorted < w_front
# boolean array - CONDITION -> aperture array at a given time step must have a value greater and less than w_front and
mask = np.any(less_than, axis=1) & np.any(~less_than, axis=1)
# aperture that respects the condition
w_tx_argsorted_masked = w_tx_argsorted[mask]
# find the first index where at each timestep the w_xt array has the w_front value
ind_w_front = np.argmax(w_tx_argsorted_masked <= w_front, axis=1)
# find front distance array
x_front = x_vert_sorted[ind_w_front]
# remove the timesteps that don't respect the condition
t_cut = t[mask]
# find the first index where at each timestep the w_xt array has the w_front value
ind_w_front = np.argmax(w_tx_argsorted <= w_front, axis=1)
# find front distance array
x_front = x_vert_sorted[ind_w_front]

# FIXME: respect the conditions for w_tx array at each timestep to filter those that have a front

# %% testing

idx = -1
w_x = w_tx[idx]
w_x_argsorted = w_x[np.argsort(x_vert)]

# %%
less_than = w_tx_argsorted < w_front
mask = np.any(less_than, axis=1) & np.any(~less_than, axis=1)
w_tx_argsorted_masked = w_tx_argsorted[mask]
# find the first index where at each timestep the w_xt array has the w_front value
ind_w_front = np.argmax(w_tx_argsorted_masked <= w_front, axis=1)
# find front distance array
x_front = x_vert_sorted[ind_w_front]
t_cut = t[mask]

# %% plotting
plt.figure()
plt.plot(t_cut, x_front, ".")
plt.show()

# %% TODO plot few aperture profiles with the w_front marked
