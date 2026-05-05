# %%
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

# %% --- Data loading -----------------------------------------------------------

type Time = np.ndarray[tuple[int], np.dtype[np.float64]]
type XPositions = np.ndarray[tuple[int], np.dtype[np.float64]]
type Field = np.ndarray[tuple[int, int], np.dtype[np.float64]]
type Parameters = dict


@dataclass
class RunData:
    t: Time  # shape: (n_time, )
    x_vert: XPositions  # shape: (n_vert, )
    w: Field  # shape: (n_time, n_vert)
    x_sc: XPositions  # shape: (n_sc, )
    sn: Field  # shape: (n_time, n_sc)
    p: Field  # shape: (n_time, n_sc)
    params: Parameters


FIELD_PATHS = {
    "x_vert": "coordinates/x_vertices",
    "t": "coordinates/t",
    "w": "fields/aperture",
    "x_sc": "coordinates/x_subcontacts",
    "sn": "fields/stress_normal",
    "p": "fields/fluid_pressure",
}


def sort_fields(
    x: XPositions,
    field: Field,
) -> tuple[XPositions, Field]:
    idx = np.argsort(x)
    return x[idx], field[:, idx]


def read_run(filepath: Path) -> RunData:
    with h5py.File(str(filepath), "r") as f:
        arrays = {k: cast(h5py.Dataset, f[path])[()] for k, path in FIELD_PATHS.items()}
        params = {
            k: cast(h5py.Dataset, f[f"parameters/{k}"])[()]
            for k in cast(h5py.Group, f["parameters"]).keys()
        }

    return RunData(**arrays, params=params)


# %% --- Front detection --------------------------------------------------------

type FrontDetectionThreshold = np.ndarray[
    tuple[int], np.dtype[np.float64]
]  # shape: (n_t, )

type TimestepHasFront = np.ndarray[tuple[int], np.dtype[np.bool_]]
type FrontPositions = np.ndarray[
    tuple[int], np.dtype[np.float64]
]  # shape: (n_t_has_front, )


def find_field_front(
    x: XPositions,
    field: Field,
    threshold: FrontDetectionThreshold,
    interpolate: bool = True,
) -> tuple[FrontPositions, TimestepHasFront]:
    """
    Find front positions where `field` crosses `threshold`. Interpolate if wished.

    Parameters
    ----------
    x          : (n_x,)      spatial coordinates
    field      : (n_t, n_x)  2D field array
    threshold  : (n_t,)      per-timestep threshold values
    interpolate: bool        whether or not to interpolate front positions

    Returns
    -------
    x_front      : (n_crossing,)  front positions (*interpolated)
    has_crossing : (n_t,) bool    mask of timesteps with a valid crossing
    """
    x, field = sort_fields(x, field)

    below = field < threshold[:, None]
    has_crossing = np.any(below, axis=1) & np.any(~below, axis=1)

    crossing = (~below[:, :-1]) & (below[:, 1:])
    cross_idx = np.argmax(crossing[has_crossing], axis=1) + 1

    if not interpolate:
        return x[cross_idx - 1], has_crossing

    field_sub = field[has_crossing]
    thresh_sub = threshold[has_crossing]
    i_left, i_right = cross_idx - 1, cross_idx

    w_left = field_sub[np.arange(len(field_sub)), i_left]
    w_right = field_sub[np.arange(len(field_sub)), i_right]
    denom = w_right - w_left

    x_front = np.full(len(field_sub), np.nan)
    valid = denom != 0
    x_front[valid] = (
        x[i_left[valid]]
        + (thresh_sub[valid] - w_left[valid])
        * (x[i_right[valid]] - x[i_left[valid]])
        / denom[valid]
    )
    return x_front, has_crossing


def find_stress_front(run: RunData, mesh_size: float = 4.0) -> tuple[np.ndarray, int]:
    positions = run.x_sc[np.argmin(run.sn, axis=1)]
    boundary = positions.max() - mesh_size
    idx = int(np.argmax(positions >= boundary))
    return positions[:idx], idx


# %% --- Physics / scaling helpers ----------------------------------------------

type HydraulicDiffusivity = float


def diffusivity(params: Parameters) -> HydraulicDiffusivity:
    return params["k_n"] * params["w_i"] ** 3 / (12 * params["mu"])


def power_law(t, A, alpha):
    return A * t**alpha


def fit_front_power_law(t: Time, x_front: XPositions) -> tuple[float, float]:
    (A, alpha), _ = curve_fit(
        power_law, t, x_front, p0=(x_front.max(), 0.8), maxfev=20_000
    )
    return A, alpha


# %% --- Front threshold definitions --------------------------------------------

type DimensionlessApertureFront = float


def self_similar_front_threshold(
    run: RunData, θ_front: DimensionlessApertureFront, is_pressure: bool = True
) -> FrontDetectionThreshold:
    """Self-similar threshold:
    pressure -  θ_front * sqrt(q0^2 * t / D) * k_n.
    aperture - w_i + θ_front * sqrt(q0^2 * t / D)."""
    D = diffusivity(run.params)
    scale = np.sqrt(run.params["q_0"] ** 2 * run.t / D)
    w_front = run.params["w_i"] + θ_front * scale
    p_front = θ_front * scale * run.params["k_n"]
    return p_front if is_pressure else w_front


type FractionalPercentage = float


def constant_aperture_threshold(
    run: RunData, pct_increase: FractionalPercentage
) -> FrontDetectionThreshold:
    return np.full_like(run.t, run.params["w_i"] * (1 + pct_increase))


type CriticalPressure = float


def constant_pressure_threshold(
    run: RunData, pc: CriticalPressure
) -> FrontDetectionThreshold:
    return np.full_like(run.t, pc)


# %% --- Case analysis ----------------------------------------------------------


# WARN: only works for pressure field
def analyse_rigid(run: RunData, threshold: FrontDetectionThreshold) -> dict:
    x_front, has_crossing = find_field_front(
        run.x_sc, run.p, threshold, interpolate=True
    )
    t_front = run.t[has_crossing]

    D = diffusivity(run.params)
    ζ_front = np.mean(x_front / np.sqrt(D * t_front))
    x_front_analytical = ζ_front * np.sqrt(D * t_front)

    prefactor, time_exponent = fit_front_power_law(t_front, x_front)

    return dict(
        t_front=t_front,
        x_front=x_front,
        x_front_analytical=x_front_analytical,
        ζ_front=ζ_front,
        prefactor=prefactor,
        time_exponent=time_exponent,
        diffusivity=D,
    )


# WARN: only works for pressure and stress field
def analyse_soft(
    run: RunData, threshold: FrontDetectionThreshold, stress_peak: bool = False
) -> dict:
    if stress_peak:
        x_front, idx = find_stress_front(run)
        t_front = run.t[:idx]
    else:
        x_front, has_crossing = find_field_front(
            run.x_sc, run.p, threshold, interpolate=True
        )
        t_front = run.t[has_crossing]

    M = run.params["k_n"] / (12.0 * run.params["mu"])
    # WARN: parameter name changed from q_0 to q
    q = run.params["q"] if "q" in run.params else run.params["q_0"]
    prefactor_analytical = (M * q**3) ** (1 / 5)
    time_exponent_analytical = 4 / 5

    # NOTE: ζ_front depends on the defined threshold
    # FIXME: zeta_front is empirical — derive analytically if possible
    ζ_front = 1.4
    x_front_analytical = (
        ζ_front * prefactor_analytical * t_front**time_exponent_analytical
    )

    prefactor_empirical, time_exponent_empirical = fit_front_power_law(t_front, x_front)

    return dict(
        t_front=t_front,
        x_front=x_front,
        x_front_analytical=x_front_analytical,
        ζ_front=ζ_front,
        prefactor_ana=prefactor_analytical,
        time_exponent_ana=time_exponent_analytical,
        prefactor_empirical=prefactor_empirical,
        time_exponent_empirical=time_exponent_empirical,
    )


# %% --- Plotting ---------------------------------------------------------------


def plot_front_rigid(ax, result: dict, title: str):
    t = result["t_front"]
    x = result["x_front"]
    ax.plot(t, x, ".", label="3DEC")
    ax.plot(
        t,
        result["x_front_analytical"],
        label=r"Analytical - $x_f = \zeta_f \, \sqrt{D t}$",
    )
    ax.text(
        t[int(0.01 * len(t))],  # pyright: ignore[reportIndexIssue]
        x[int(0.8 * len(x))],  # pyright: ignore[reportIndexIssue]
        rf"$\zeta_f={result['ζ_front']:.2f}$, $D={result['diffusivity']:.2f}~\mathrm{{m^2/s}}$",
    )
    ax.annotate(
        text="finite-size effect",
        xy=(t[int(0.9 * len(t))], x[int(0.9 * len(x))]),
        xytext=(t[int(0.6 * len(t))], x[int(0.99 * len(x))]),
        arrowprops=dict(arrowstyle="->", lw=2),
    )
    ax.set_xlabel(r"Time $t$, [s]")
    ax.set_ylabel("Front distance $x_f$, [m]")
    ax.set_title(title)
    ax.legend()


def plot_front_soft(ax, result: dict, title: str):
    t = result["t_front"]
    x = result["x_front"]
    ax.plot(t, x, ".", label="3DEC")
    ax.plot(
        t,
        result["x_front_analytical"],
        label=r"Analytical - $x_f = \zeta_f \, A \, t^{4/5}$",
    )
    ax.text(
        t[int(0.01 * len(t))],  # pyright: ignore[reportIndexIssue]
        x[int(0.65 * len(x))],  # pyright: ignore[reportIndexIssue]
        rf"$A=\left( \frac{{k_n q_0^3}}{{12 \mu}} \right)^{{1/5}}={result['prefactor_ana']:.2f}~\mathrm{{m/s^{{4/5}}}}$",
    )
    ax.text(
        t[int(0.01 * len(t))],  # pyright: ignore[reportIndexIssue]
        x[int(0.5 * len(x))],  # pyright: ignore[reportIndexIssue]
        rf"$\zeta_f={result['ζ_front']:.2f}$",
    )
    ax.set_xlabel(r"Time $t$, [s]")
    ax.set_ylabel("Front distance $x_f$, [m]")
    ax.set_title(title)
    ax.legend()


# TODO: add intermediate case where physics transitions from rigid to soft behavior

# %% --- Main -------------------------------------------------------------------

result_dir = Path.cwd() / "results" / "3dec" / "runs"

# Rigid case
run_rigid = read_run(result_dir / "run-q-1e-06.hdf5")
theta_front = 0.1
threshold = self_similar_front_threshold(run_rigid, theta_front, is_pressure=True)
rigid = analyse_rigid(run_rigid, threshold)

# Soft case
run_soft = read_run(result_dir / "run-q-1e-03.hdf5")
# INFO: critical pressure influences results
pc = 5e5
threshold = constant_pressure_threshold(run_soft, pc=pc)
soft = analyse_soft(run_soft, threshold=threshold, stress_peak=False)

# %% --- Plotting ------

figure_dir = Path.cwd() / "figures"

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
plot_front_rigid(
    ax=ax1,
    result=rigid,
    title=rf"Rigid Case — $q_0={run_rigid.params['q_0']}~\mathrm{{m^2/s}}$",
)

plot_front_soft(
    ax=ax2,
    result=soft,
    title=rf"Soft Case — $q_0={run_soft.params['q_0']}~\mathrm{{m^2/s}}$",
)
fig.savefig(figure_dir / "front.png", dpi=200)
plt.show()


# %% --- Testing --------
