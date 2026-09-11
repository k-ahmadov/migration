"""Locate propagation fronts in space-time fields, and threshold definitions."""

import numpy as np

from fracinj import io, physics
from fracinj.types import BoolVector, Field, Vector


def find_field_front(
    x: Vector,
    field: Field,
    threshold: Vector,
    interpolate: bool = True,
) -> tuple[Vector, BoolVector]:
    """Front positions where ``field`` first drops below ``threshold``.

    Parameters
    ----------
    x          : (n_x,)      spatial coordinates
    field      : (n_t, n_x)  space-time field
    threshold  : (n_t,)      per-timestep threshold
    interpolate : linearly interpolate the crossing between nodes

    Returns
    -------
    x_front      : (n_crossing,)  front positions
    has_crossing : (n_t,) bool    timesteps that contain a crossing
    """
    x, field = io.sort_fields(x, field)

    below = field < threshold[:, None]
    has_crossing = np.any(below, axis=1) & np.any(~below, axis=1)

    crossing = (~below[:, :-1]) & (below[:, 1:])
    cross_idx = np.argmax(crossing[has_crossing], axis=1) + 1

    if not interpolate:
        return x[cross_idx - 1], has_crossing

    field_sub = field[has_crossing]
    thresh_sub = threshold[has_crossing]
    i_left, i_right = cross_idx - 1, cross_idx
    rows = np.arange(len(field_sub))

    w_left = field_sub[rows, i_left]
    w_right = field_sub[rows, i_right]
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


def find_stress_front(x: Vector, sn: Field, L: float) -> tuple[Vector, int]:
    """Track the most-tensile point of each stress profile until it reaches ``L``.

    Returns the front positions up to (excluding) the timestep where the
    front first comes within one mesh cell of ``L``, and that timestep's index.
    """
    if sn.shape[0] == 0:
        raise ValueError("sn must contain at least one timestep")

    positions = x[np.argmin(sn, axis=1)]
    boundary = L - (x[-1] - x[-2])

    reached = np.flatnonzero(positions >= boundary)
    idx = int(reached[0]) if reached.size else len(positions)
    return positions[:idx], idx


# --- Front threshold definitions --------------------------------------------


def self_similar_threshold(
    run: io.RunData, theta_front: float, *, pressure: bool = True
) -> Vector:
    """Self-similar threshold that grows as ``sqrt(q0**2 t / D)``.

    ``pressure``: return the pressure threshold, else the aperture threshold.
    """
    scale = theta_front * np.sqrt(
        run.params.q_0**2 * run.t / physics.diffusivity(run.params)
    )
    return scale * run.params.k_n if pressure else run.params.w_i + scale


def constant_aperture_threshold(run: io.RunData, pct_increase: float) -> Vector:
    return np.full_like(run.t, run.params.w_i * (1 + pct_increase))


def constant_pressure_threshold(t: Vector, pc: float) -> Vector:
    return np.full_like(t, pc)


# --- Seismic front based on percentile --------------------------------------------


def percentile_envelope(t, distance, bin_size, percentile):
    "Chooses the distance based on the given percentile per bin of seismic data"
    # number of bins based on bin size
    n_bins = len(distance) // bin_size
    n_trim = n_bins * bin_size
    # trim distance and time data per bin
    d_bins = distance[:n_trim].reshape(n_bins, bin_size)
    t_bins = t[:n_trim].reshape(n_bins, bin_size)
    # choose distance value from each bin according to the given percentile
    p_vals = np.percentile(d_bins, percentile, axis=1, method="nearest")
    # find its indices
    idx = np.argmin(np.abs(d_bins - p_vals[:, None]), axis=1)
    # import the corresponding time and distance values
    t_pct = t_bins[np.arange(n_bins), idx]
    d_pct = d_bins[np.arange(n_bins), idx]
    return t_pct, d_pct


def seismic_envelope(
    t_data: np.ndarray,
    d_data: np.ndarray,
    percentile: float = 90,
    min_events: int = 20,
    max_dt: float = 0.1 * 86400,
    n_eval: int = 1000,
    t_eval: np.ndarray | None = None,
):
    """
    Adaptive time-window envelope.

    For each evaluation time, use a backward-looking window that is
    widened in time until it contains at least `min_events` events
    (or capped at `max_dt`), then take the percentile of distance
    within that window.

    Assumes t_s is sorted ascending.
    """
    t_data = np.asarray(t_data)
    d_data = np.asarray(d_data)

    if t_eval is None:
        t_eval = np.linspace(t_data.min(), t_data.max(), n_eval)

    if max_dt is None:
        max_dt = t_data.max() - t_data.min()  # effectively no cap

    d_env = np.full(len(t_eval), np.nan)
    t_env = np.full(len(t_eval), np.nan)

    # searchsorted indices let us binary-search window edges fast
    for i, t_now in enumerate(t_eval):
        # right edge: last event at or before t_now
        right = np.searchsorted(t_data, t_now, side="right")
        if right == 0:
            continue  # no events yet

        # find left edge: smallest window (in time, capped at max_dt)
        # containing >= min_events events, searching backward
        t_left_cap = t_now - max_dt
        left_cap_idx = np.searchsorted(t_data, t_left_cap, side="left")

        left = right - min_events
        left = max(left, left_cap_idx, 0)

        window_d = d_data[left:right]

        if len(window_d) == 0:
            continue

        d_e = np.percentile(window_d, percentile)
        d_env[i] = d_e
        t_env[i] = t_now  # use evaluation time directly, not argmin matching

    valid = ~np.isnan(d_env)
    return t_env[valid], d_env[valid]
