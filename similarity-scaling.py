# %%
from pathlib import Path
from typing import cast

import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

# %%


def find_pressure_front(
    p_tx: np.ndarray, Pc: float, t_points: np.ndarray, x: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    # Pressure front = first position from the right where pressure is >= 1 bar overpressure
    # Boolean mask: pressure above threshold for each (time, space)
    above_Pc = p_tx >= Pc  # shape: (Nt, Nx)

    # A "front" exists when the row is neither entirely below Pc nor entirely above Pc
    row_has_any_above = above_Pc.any(axis=1)  # at least one row with a cell >= Pc
    row_all_above = above_Pc.all(axis=1)  # all cells >= Pc
    has_front = row_has_any_above & ~row_all_above

    # For times with a front: find the rightmost x-index where pressure is >= Pc
    # Reverse x so "rightmost" becomes "leftmost" in the reversed array.
    reverse = above_Pc[has_front, ::-1]
    dist_from_right = reverse.argmax(axis=1)  # first True in reversed row
    front_x_idx = (p_tx.shape[1] - 1) - dist_from_right

    # Map indices to coordinates
    front_t_idx = np.flatnonzero(has_front)
    t_front = t_points[front_t_idx]
    front = x[front_x_idx]

    return t_front, front


def power_law(t, A, alpha):
    return A * t**alpha


def fit_power_law_to_front(
    t_points: np.ndarray, front: np.ndarray, power_law=power_law
) -> tuple[float, float]:

    (A, alpha), _ = curve_fit(power_law, t_points, front)
    return float(A), float(alpha)


def fit_power_law_fixed_alpha(
    t_points: np.ndarray,
    front: np.ndarray,
    alpha_fixed: float,
) -> float:

    def model(t, A):
        return A * t**alpha_fixed

    (A,), _ = curve_fit(model, t_points, front)
    return float(A)


# %%
# -----------------------------------------------------------------------------
# Load 3DEC output data
# -----------------------------------------------------------------------------

filepath = Path.cwd() / "results" / "3dec" / "runs" / "run-q-1e-04.hdf5"

with h5py.File(filepath, "r") as f:
    x = cast(h5py.Dataset, f["coordinates/x_vertices"])[:]
    x_sc = cast(h5py.Dataset, f["coordinates/x_subcontacts"])[:]
    w_tx = cast(h5py.Dataset, f["fields/aperture"])[:]
    p_tx = cast(h5py.Dataset, f["fields/fluid_pressure"])[:]
    t_points = cast(h5py.Dataset, f["coordinates/t"])[:]
    params = {
        k: cast(h5py.Dataset, f[f"parameters/{k}"])[()]
        for k in cast(h5py.Group, f["parameters"]).keys()
    }

# %%
a = params["k_n"] / (12 * params["mu"])
t_c = a * params["w_i"] ** 5 / params["q_0"] ** 2
x_sorted = np.sort(x)
ind_sort_x = np.argsort(x)
D = a * params["w_i"] ** 3
print(t_c)
# %%
# ---------------------
# Early-time
# ---------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3), constrained_layout=True)
for i in np.linspace(6, 51, 5, dtype=np.int16):
    w_x_sorted = w_tx[i, ind_sort_x]
    ax1.plot(x_sorted, w_x_sorted * 1e3, label=f"t={t_points[i]:.2f} s")
    theta = (w_x_sorted - params["w_i"]) / (params["q_0"] ** 2 * t_points[i] / D) ** (
        1 / 2
    )
    zeta = x_sorted / (D * t_points[i]) ** (1 / 2)
    ax2.plot(zeta, theta)
ax1.set_xlabel("x, [m]")
ax1.set_ylabel("w, [mm]")
ax1.set_xlim(-2, 40)
ax1.legend()
ax1.text(0.6, 0.3, rf"$t_c=${t_c:.2f} s", transform=ax1.transAxes)

ax2.set_xlabel(r"$\zeta = x/(D t)^{1/2}$")
ax2.set_ylabel(r"$\theta = (w-w_i)/(q_0^2 t/ D)^{1/2}$")
ax2.set_xlim(-0.2, 8)
plt.suptitle(r"Early-time solution, $\alpha=1/2$")
# plt.show()
# plt.savefig("./figures/early-time-similarity.png")
plt.close(fig)

# %%
# ---------------------
# Late-time
# ---------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3), constrained_layout=True)
for i in np.linspace(601, len(t_points) - 10, 5, dtype=np.int16):
    w_x_sorted = w_tx[i, ind_sort_x]
    ax1.plot(x_sorted, w_x_sorted * 1e3, label=f"t={t_points[i]:.2f} s")
    theta = (w_x_sorted) / (params["q_0"] ** 2 * t_points[i] / a) ** (1 / 5)
    zeta = x_sorted / (a * params["q_0"] ** 3 * t_points[i] ** 4) ** (1 / 5)
    ax2.plot(zeta, theta)
ax1.set_xlabel("x, [m]")
ax1.set_ylabel("w, [mm]")
ax1.legend()
ax1.text(0.75, 0.35, rf"$t_c=${t_c:.2f} s", transform=ax1.transAxes)

ax2.set_xlabel(r"$\zeta = x/(a q_0^3 t^4)^{4/5}$")
ax2.set_ylabel(r"$\theta = w / (q_0^2 t / a)^{1/5}$")
plt.suptitle(r"Late-time solution, $\alpha=4/5$")
# plt.show()
plt.savefig("./figures/late-time-similarity.png")
plt.close(fig)

# %%
# ---------------------
# Front migration
# ---------------------
Pc = 1e5
x_sc_sorted = np.sort(x_sc)
p_tx_sorted = p_tx[:, np.argsort(x_sc)]
t_front, front = find_pressure_front(p_tx_sorted, Pc, t_points, x_sc_sorted)
t_front_early_time, front_early_time = find_pressure_front(
    p_tx_sorted[:100, :], Pc, t_points[:100], x_sc_sorted
)
t_front_late_time, front_late_time = find_pressure_front(
    p_tx_sorted[600:, :], Pc, t_points[600:], x_sc_sorted
)

A, alpha = fit_power_law_to_front(t_front, front)

alpha_late_time = 0.8
A_late_time = fit_power_law_fixed_alpha(
    t_front_late_time, front_late_time, alpha_late_time
)

alpha_early_time = 0.5
A_early_time = fit_power_law_fixed_alpha(
    t_front_early_time, front_early_time, alpha_early_time
)

# %%
fig, ax = plt.subplots()

ax.plot(
    t_front, front, ".", label="3DEC, position of 1 bar overpressure", color="tab:blue"
)
ax.plot(
    t_front, A * t_front**alpha, "-", label=rf"Fit: $A=${A:.2f}, $\alpha=${alpha:.2f}"
)
ax.plot(
    t_front,
    A_early_time * t_front**alpha_early_time,
    label=r"Early-time scaling, $\alpha=1/2$",
)
ax.plot(
    t_front,
    A_late_time * t_front**alpha_late_time,
    label=r"Late-time scaling, $\alpha=4/5$",
)
ax.set_xlabel("Time, [s]")
ax.set_ylabel("Pressure front position, [m]")
ax.set_title(f"Applied injection rate, $q_0=${params['q_0']:.0e} m$^2$/s")
ax.legend()
plt.show()
# plt.savefig("./figures/early-and-late-time-front-migration.png")
plt.close(fig)

# %%

plt.plot(t_points[1:], w_tx[1:, np.argsort(x)][:, 0], ".")
plt.show()

print(w_tx[:, np.argsort(x)][:, 0].shape)

print(w_tx.shape)
print(t_points.shape)
# %%
# Pc = 1e6
# t_front, front = find_front(p_tx, Pc, t_points, x)
# A, alpha = fit_power_law_to_front(t_front, front)
# print(A, alpha)
