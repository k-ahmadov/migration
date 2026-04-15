# %% [markdown]
# ## Analysis of the Coulomb Failure Function
#
# Here, I extract fluid pressure, normal stress, and shear stress along the fracture
# as functions of space and time, and use them to compute the
# Coulomb Failure Function (CFF).
#
# No failure criterion is enforced in the 3DEC model; the CFF is therefore used
# purely as a diagnostic indicator of proximity to failure.
#
# ---
#
# ### Coulomb Failure Function
#
# The Coulomb Failure Function is defined as:
#
# $$
# \mathrm{CFF}(x,t) = \tau(x,t) - \mu \left(\sigma_n(x,t) - P(x,t)\right),
# $$
#
# Interpretation:
# - **CFF < 0**: stable conditions
# - **CFF = 0**: onset of failure
# - **CFF > 0**: failure condition
#
# ---
#
# ### Contribution of each variable to the CFF
#
# To assess which physical mechanism dominates the approach to failure,
# I compute dimensionless contribution factors for shear stress,
# fluid pressure, and normal stress.
#
# The shear stress contribution is defined as:
#
# $$
# \mathrm{SSC} = \frac{\tau(x,t_i)}{\mathrm{CFF}(x,t_i)}.
# $$
#
# If SSC is close to 1, failure is primarily driven by shear stress.
#
# Similarly, the fluid pressure and normal stress contributions are:
#
# $$
# \mathrm{FPC} = \frac{\mu P(x,t_i)}{\mathrm{CFF}(x,t_i)}, \quad
# \mathrm{NSC} = \frac{\mu \sigma_n(x,t_i)}{\mathrm{CFF}(x,t_i)}.
# $$
#
# ---
#
# $$
# \begin{array}{ll}
# \textbf{Property} & \textbf{Value} \\
# \hline
# \text{Fracture length, } L & 100~\text{m} \\
# \text{Normal stiffness, } k_n & 50~\text{GPa/m} \\
# \text{Initial aperture, } w_i & 0.1~\text{mm} \\
# \text{Fluid viscosity, } \mu_w & 10^{-3}~\text{Pa}\cdot\text{s} \\
# \text{Duration, } T & 100~\text{s} \\
# \text{Friction, } \mu & 0.6 \\
# \text{Young's modulus, } E & 60~\text{GPa} \\
# \text{Poisson's ratio, } \nu & 0.25 \\
# \end{array}
# $$

# %%
from pathlib import Path
from typing import cast

import h5py
import matplotlib.pyplot as plt
import numpy as np

# %%
# -----------------------------------------------------------------------------
# Load 3DEC output data
# -----------------------------------------------------------------------------

filepath = Path.cwd() / "results" / "3dec" / "runs" / "run-q-1e-03.hdf5"

with h5py.File(filepath, "r") as f:
    x_subcontacts = cast(h5py.Dataset, f["coordinates/x_subcontacts"])[:]
    x_vertices = cast(h5py.Dataset, f["coordinates/x_vertices"])[:]
    w_tx = cast(h5py.Dataset, f["fields/aperture"])[:]
    p_tx = cast(h5py.Dataset, f["fields/fluid_pressure"])[:]
    sn_tx = cast(h5py.Dataset, f["fields/stress_normal"])[:]
    tau_tx = cast(h5py.Dataset, f["fields/stress_shear"])[:]
    t_points = cast(h5py.Dataset, f["coordinates/t"])[:]

# %%
# -----------------------------------------------------------------------------
# Select time for spatial analysis
# -----------------------------------------------------------------------------

t = 1.0  # target time [s]
idx_t = np.searchsorted(t_points, t)

p_x = p_tx[idx_t]
sn_x = sn_tx[idx_t]
tau_x = tau_tx[idx_t]
w_x = w_tx[idx_t]
print(p_x)
# %%
# -----------------------------------------------------------------------------
# Plot pressure and stresses along the fracture
# -----------------------------------------------------------------------------

fig, ax = plt.subplots()

ax.plot(x_subcontacts, p_x, "o", label="fluid pressure")
ax.plot(x_subcontacts, -sn_x, "^", label="tensile normal stress")
# ax.plot(x_subcontacts, tau_x, "x", label="shear stress")

ax.set_yscale("log")
plt.show()

# # Create inset (position in axes coordinates 0–1)
# axins = ax.inset_axes((0.5, 0.35, 0.4, 0.4))
# # Same data
# axins.plot(x_subcontacts, p_x, "o")
# axins.plot(x_subcontacts, -sn_x, "^")
# axins.plot(x_subcontacts, tau_x, "x")
# # Zoom region (DATA coordinates)
# axins.set_xlim(40, 65)
# axins.set_ylim(-2e5, 2e5)
# # disable axis ticks
# axins.set_xticklabels([])
# axins.set_yticklabels([])
# # connectors indicating zoom region
# inset_ind = ax.indicate_inset_zoom(axins, edgecolor="black")
# inset_ind.connectors[0].set_visible(False)  # pyright: ignore
# inset_ind.connectors[2].set_visible(True)  # pyright: ignore

ax.set_xlabel("x [m]")
ax.set_ylabel(r"$P(x)$ / $\sigma_n(x)$ / $\tau(x)$ [Pa]")
ax.legend()
# plt.savefig("./figures/pressure-normal-shear-stresses.png")
plt.show()

# %%
# -----------------------------------------------------------------------------
# Compute and plot Coulomb Failure Function
# -----------------------------------------------------------------------------

mu = 0.6

CFF = tau_x - mu * (sn_x - p_x)

fig, ax = plt.subplots()
ax.plot(x_subcontacts, CFF, "o", label=r"$\tau - \mu (\sigma_n - P)$")
ax.plot(x_subcontacts, mu * p_x, "o", label=r"$\mu P$")
ax.set_xlabel("x [m]")
ax.set_ylabel("Coulomb Failure Function [Pa]")
ax.legend()
ax.set_yscale("log")
# plt.savefig("./figures/CFF.png")
plt.show()

# %%
# -----------------------------------------------------------------------------
# Contribution analysis
# -----------------------------------------------------------------------------

SSC = tau_x / CFF
FPC = mu * p_x / CFF
NSC = -mu * sn_x / CFF

fig, ax = plt.subplots()
ax.plot(x_subcontacts, FPC, "o", label="fluid pressure contribution")
ax.plot(x_subcontacts, NSC, "^", label="normal stress contribution")
ax.plot(x_subcontacts, SSC, "x", label="shear stress contribution")
ax.axvspan(60, x_subcontacts.max(), color="gray", alpha=0.2, label="noisy region?")
ax.set_ylim((-0.2, 1.2))
ax.set_xlabel("x [m]")
ax.set_ylabel("Relative contribution [-]")
ax.legend()
# plt.savefig("./figures/contribution-analysis.png")
plt.show()


# %%
