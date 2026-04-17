# %%
from pathlib import Path
from typing import cast

import h5py
import matplotlib.pyplot as plt
import numpy as np

from mysolvers.exact_solutions import solve_linear_diffusion_const_flux

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
#
# # sort the distance array
# x_vert_sorted = np.sort(x_vert)
# # sort each row (timestep) of w_tx according to the sorted distance
# w_tx_argsorted = w_tx[:, np.argsort(x_vert)]
# # value of aperture to track as the front
# w_front = params["w_i"] * (1 + 0.01)
# # boolean array - if aperture array is less than the chosen w_front
# less_than = w_tx_argsorted < w_front
# # boolean array - CONDITION -> aperture array at a given time step must have a value greater and less than w_front and
# mask = np.any(less_than, axis=1) & np.any(~less_than, axis=1)
# # aperture that respects the condition
# w_tx_argsorted_masked = w_tx_argsorted[mask]
# # find the first index where at each timestep the w_xt array has the w_front value
# ind_w_front = np.argmax(w_tx_argsorted_masked < w_front, axis=1)
# # find front distance array
# x_front = x_vert_sorted[ind_w_front]
# # remove the timesteps that don't respect the condition
# t_front = t[mask]

# %% FIXME: the front definition is not correct in similarity form (it changes with time)
# sort once, consistently
sort_idx = np.argsort(x_vert)
x_vert_sorted = x_vert[sort_idx]
w_tx_sorted = w_tx[:, sort_idx]

# define front threshold
w_front = params["w_i"] * (1 + 0.01)

# identify valid timesteps (crossing exists)
below = w_tx_sorted < w_front
above = ~below
mask = np.any(below, axis=1) & np.any(above, axis=1)

# apply mask
w_valid = w_tx_sorted[mask]
t_front = t[mask]

# find first index where it goes from >= w_front to < w_front
cross_idx = np.argmax(below[mask], axis=1)

# avoid edge issues (ensure we have a left neighbor)
cross_idx = np.clip(cross_idx, 1, len(x_vert_sorted) - 1)

# indices for interpolation
i_left = cross_idx - 1
i_right = cross_idx

# gather values
x_left = x_vert_sorted[i_left]
x_right = x_vert_sorted[i_right]

w_left = w_valid[np.arange(len(w_valid)), i_left]
w_right = w_valid[np.arange(len(w_valid)), i_right]

# linear interpolation:
# x_f = x_left + (w_front - w_left) * (x_right - x_left) / (w_right - w_left)
denom = w_right - w_left

# avoid division by zero (flat segment edge case)
denom[denom == 0] = np.nan

x_front = x_left + (w_front - w_left) * (x_right - x_left) / denom

# %% FIXME: add the prefactor to the analytical solution for front
M = params["k_n"] / (12 * params["mu"])
D = M * params["w_i"] ** 3
x_front_analytical = np.sqrt(D * t_front)
zeta_analytical = np.linspace(0, 10, 100)
theta_analytical = solve_linear_diffusion_const_flux(zeta_analytical)


# %% plot aperture profiles and the front point
plt.figure()
for i in [100, 500, 1000]:
    plt.plot(x_vert_sorted, w_valid[i], ".", label=f"$t_i=${t_front[i]:.1f} s")
    plt.plot(x_front[i], w_front, "kx")
plt.legend()
plt.show()

# %% compare analytical and 3dec in dimensionless space and mark the aperture point
plt.figure()
for i in [100, 500, 1000, 2000]:
    theta = (w_valid[i] - params["w_i"]) / np.sqrt(params["q_0"] ** 2 * t_front[i] / D)
    zeta = x_vert_sorted / np.sqrt(D * t_front[i])
    plt.plot(zeta, theta, ".", label=f"$t_i=${t_front[i]:.1f} s")

    theta_front = (w_front - params["w_i"]) / np.sqrt(
        params["q_0"] ** 2 * t_front[i] / D
    )
    zeta_front = x_front[i] / np.sqrt(D * t_front[i])
    plt.plot(zeta_front, theta_front, "kx")

plt.plot(zeta_analytical, theta_analytical, "k", label="analytical")
plt.legend()
plt.show()

# %% plotting
plt.figure()
plt.plot(t_front, x_front, ".", label="3DEC")
plt.plot(t_front, x_front_analytical, label=r"$x_f = \sqrt{D t}$")
plt.legend()
plt.show()

# %% testing

idx = -1
w_x = w_tx[idx]
w_x_argsorted = w_x[np.argsort(x_vert)]

# %%
# %% plotting
plt.figure()
plt.plot(t_front, x_front, ".")
plt.show()

# %% TODO plot few aperture profiles with the w_front marked
