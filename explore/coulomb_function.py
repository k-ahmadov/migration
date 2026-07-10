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

import matplotlib.pyplot as plt
import numpy as np

from mypackages import file_io


# %%
def zoom_region(ax, run: file_io.RunData):
    # Create inset (position in axes coordinates 0–1)
    axins = ax.inset_axes((0.5, 0.35, 0.4, 0.4))
    # Same data
    axins.plot(run.x_sc, run.p, "o")
    axins.plot(run.x_sc, -run.sn, "^")
    axins.plot(run.x_sc, run.tau, "x")
    # Zoom region (DATA coordinates)
    axins.set_xlim(40, 65)
    axins.set_ylim(-2e5, 2e5)
    # disable axis ticks
    axins.set_xticklabels([])
    axins.set_yticklabels([])
    # connectors indicating zoom region
    inset_ind = ax.indicate_inset_zoom(axins, edgecolor="black")
    inset_ind.connectors[0].set_visible(False)  # pyright: ignore
    inset_ind.connectors[2].set_visible(True)  # pyright: ignore


# %%
# -----------------------------------------------------------------------------
# Load 3DEC output data
# -----------------------------------------------------------------------------

filepath = Path.cwd() / "results" / "3dec" / "runs" / "run-q-1e-03.hdf5"

run = file_io.read_run(filepath=filepath)
assert run.tau
# %% Select time for spatial analysis
t_i = 5.0  # target time [s]
idx_t = np.searchsorted(run.t, t_i)

# %%
# -----------------------------------------------------------------------------
# Plot pressure and stresses along the fracture
# -----------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(6.4 / 1.5, 4.8 / 1.5), dpi=200, layout="tight")

ax.plot(run.x_sc, run.tau[idx_t], "+", color="tab:gray", label=r"$\tau_s$")
ax.plot(run.x_sc, run.p[idx_t], "o", color="tab:blue", label=r"$p$")
# ax.plot(
#     run.x_sc, run.sn[idx_t], "x", color="tab:red", label=r"$\sigma_n$"
# )
ax.plot(run.x_sc, -run.sn[idx_t], "^", color="tab:orange", label=r"$-\sigma_n$")
ax.set(xlabel="x [m]", ylabel=r"Stress [Pa]")
leg = ax.legend()
leg.set_draggable(True)
# plt.savefig("./figures/pressure-normal-shear-stresses.png")
plt.show()

# %%
# -----------------------------------------------------------------------------
# Compute and plot Coulomb Failure Function
# -----------------------------------------------------------------------------

mu = 0.6

CFF = run.tau[idx_t] - mu * (run.sn[idx_t] - run.p[idx_t])

fig, ax = plt.subplots(figsize=(6.4 / 1.5, 4.8 / 1.5), dpi=200, layout="tight")
ax.plot(run.x_sc, mu * run.p[idx_t], "o", color="tab:blue", label=r"$\mu P$")
ax.plot(run.x_sc, CFF, "x", color="tab:red", label=r"$\tau_s - \mu (\sigma_n - P)$")
ax.legend()
ax.set(yscale="log", xlabel="Distance [m]", ylabel=r"Stress [Pa]")
# plt.savefig("./figures/CFF.png")
plt.show()

# %%
# -----------------------------------------------------------------------------
# Contribution analysis
# -----------------------------------------------------------------------------

SSC = run.tau[idx_t] / CFF
FPC = mu * run.p[idx_t] / CFF
NSC = -mu * run.sn[idx_t] / CFF

fig, ax = plt.subplots(figsize=(6.4 / 1.3, 4.8 / 1.3), dpi=200)
ax.plot(run.x_sc, FPC, "o", color="tab:blue", label="cotribution of $p$")
ax.plot(run.x_sc, NSC, "^", color="tab:orange", label="contribution of $\\sigma_n$")
# ax.plot(run.x_sc, SSC, "+", color="tab:gray", label="shear stress contribution")
# ax.axvspan(60, run.x_sc.max(), color="gray", alpha=0.2, label="noisy region?")
ax.set(
    ylim=(-0.2, 1.2),
    xlim=(-1, 100),
    xlabel="Distance [m]",
    ylabel=r"Contribution to Coulomb[-]",
)
ax.legend()
plt.tight_layout()
# plt.savefig("./figures/contribution-analysis.png")
plt.show()


# %%
