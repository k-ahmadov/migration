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
    x_front[valid] = x[i_left[valid]] + (thresh_sub[valid] - w_left[valid]) * (
        x[i_right[valid]] - x[i_left[valid]]
    ) / denom[valid]
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
    scale = theta_front * np.sqrt(run.params.q_0**2 * run.t / physics.diffusivity(run.params))
    return scale * run.params.k_n if pressure else run.params.w_i + scale


def constant_aperture_threshold(run: io.RunData, pct_increase: float) -> Vector:
    return np.full_like(run.t, run.params.w_i * (1 + pct_increase))


def constant_pressure_threshold(t: Vector, pc: float) -> Vector:
    return np.full_like(t, pc)
