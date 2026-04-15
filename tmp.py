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

M = params["k_n"] / (12 * params["mu"])
zeta_ana = np.linspace(0, 10, 100)
theta_ana = solve_linear_diffusion_const_flux(zeta_ana)
idx = 800
w_x = w_tx[idx]
sn_x = sn_tx[idx]

# fig, ax = plt.subplots()
#
# ax.plot(x_vert, w_x, ".", label=f"t={t[idx]:.2f} s")
# # ax.plot(x_sc, -sn_x, ".", label=f"t={t[idx]:.2f} s")
# ax.legend()
# plt.show()


# %%
fig, ax = plt.subplots()

for i in [50, 200, 500, 1000, -1]:
    w_x = w_tx[i]
    # ax.plot(x_vert, w_x, '.')
    t_i = t[i]

    theta = (w_x - params["w_i"]) / np.sqrt(params["q_0"] ** 2 * t_i / M)
    zeta = x_vert / np.sqrt(M * params["w_i"] ** 3 * t_i)

    ax.plot(zeta, theta / 1e6, ".")

ax.plot(zeta_ana, theta_ana)
plt.show()

# %%
