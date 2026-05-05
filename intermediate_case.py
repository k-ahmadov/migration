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


# %% --- Physics / scaling helpers ----------------------------------------------

type HydraulicDiffusivity = float


def diffusivity(params: Parameters) -> HydraulicDiffusivity:
    return params["k_n"] * params["w_i"] ** 3 / (12 * params["mu"])


type ScalingExponent = float
type Prefactor = float


def power_law(t: Time, A: Prefactor, α: ScalingExponent):
    return A * t**α


def fit_front_power_law(
    t: Time, x_front: XPositions, α: ScalingExponent = 0.0
) -> tuple[Prefactor, ScalingExponent]:
    if α != 0.0:
        (A,), _ = curve_fit(lambda t, A: A * t**α, t, x_front, p0=(x_front.max(),))
        return A
    (A, α), _ = curve_fit(power_law, t, x_front, p0=(x_front.max(), 0.8), maxfev=20_000)
    return A, α


type CharacteristicTime = float


def characteristic_time(run: RunData) -> CharacteristicTime:
    M = run.params["k_n"] / (12 * run.params["mu"])
    t_c = M * run.params["w_i"] ** 5 / run.params["q_0"] ** 2
    return t_c


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
    t: Time, pc: CriticalPressure
) -> FrontDetectionThreshold:
    return np.full_like(t, pc)


# %% --- Case analysis ----------------------------------------------------

# TODO:: shorten the code


def analyze_general(run: RunData, pc: CriticalPressure) -> dict:
    threshold = constant_pressure_threshold(run.t, pc)
    x_front, has_crossing = find_field_front(
        run.x_sc, run.p, threshold, interpolate=True
    )
    t_front = run.t[has_crossing]
    A_emp, α_emp = fit_front_power_law(t_front, x_front)
    return dict(
        threshold=threshold,
        x_front=x_front,
        t_front=t_front,
        has_front=has_crossing,
        A_emp=A_emp,
        α_emp=α_emp,
    )


def analyze_rigid(run: RunData, pc: CriticalPressure):
    t_c = characteristic_time(run)
    idx_t_c = np.searchsorted(run.t, t_c)
    threshold = constant_pressure_threshold(run.t[:idx_t_c], pc)
    x_front, has_crossing = find_field_front(
        run.x_sc, run.p[:idx_t_c], threshold, interpolate=True
    )
    t_front = run.t[:idx_t_c][has_crossing]
    D = diffusivity(run.params)
    α_ana = 1 / 2
    # exclude first index where t_front=0
    ζ_front = np.mean(x_front[1:] / (D * t_front[1:]) ** (α_ana))
    A_ana = ζ_front * D ** (α_ana)
    A_emp, α_emp = fit_front_power_law(t_front, x_front)
    return dict(
        threshold=threshold,
        x_front=x_front,
        t_front=t_front,
        has_front=has_crossing,
        A_emp=A_emp,
        α_emp=α_emp,
        A_ana=A_ana,
        α_ana=α_ana,
    )


def analyze_soft(run: RunData, pc: CriticalPressure):
    t_c = characteristic_time(run)
    idx_t_c = np.searchsorted(run.t, t_c)
    threshold = constant_pressure_threshold(run.t[idx_t_c * 10 :], pc)
    x_front, has_crossing = find_field_front(
        run.x_sc, run.p[idx_t_c * 10 :], threshold, interpolate=True
    )
    t_front = run.t[idx_t_c * 10 :][has_crossing]

    M = run.params["k_n"] / (12.0 * run.params["mu"])
    q = run.params["q"] if "q" in run.params else run.params["q_0"]
    α_ana = 4 / 5
    C_ana = (M * q**3) ** (1 / 5)
    ζ_front = 2.15
    A_ana = ζ_front * C_ana

    A_emp, α_emp = fit_front_power_law(t_front, x_front)
    return dict(
        threshold=threshold,
        x_front=x_front,
        t_front=t_front,
        has_front=has_crossing,
        A_emp=A_emp,
        α_emp=α_emp,
        A_ana=A_ana,
        α_ana=α_ana,
    )


# %% --- Main ------------------------------

result_dir = Path.cwd() / "results" / "3dec" / "runs"

run = read_run(result_dir / "run-q-1e-04.hdf5")

pc = 1e5
result_general = analyze_general(run, pc)
result_early_time = analyze_rigid(run, pc)
result_late_time = analyze_soft(run, pc)


# %% --- Plotting ----

plt.figure()
plt.plot(result_general["t_front"], result_general["x_front"], ".")
plt.plot(
    result_general["t_front"],
    result_early_time["A_ana"]
    * result_general["t_front"] ** (result_early_time["α_ana"]),
    "-",
)
plt.plot(
    result_general["t_front"],
    result_late_time["A_ana"] * result_general["t_front"] ** result_late_time["α_ana"],
    "-",
)
plt.show()
