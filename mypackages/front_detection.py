import numpy as np

from mypackages import file_io
from mypackages.physics import diffusivity
from mypackages.types import (
    CriticalPressure,
    DimensionlessApertureAtFront,
    Field,
    FractionalPercentage,
    FrontDetectionThreshold,
    FrontPositions,
    Time,
    TimestepHasFront,
    XPositions,
)


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
    x, field = file_io.sort_fields(x, field)

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


def find_stress_front(x, sn, mesh_size: float = 4.0) -> tuple[np.ndarray, int]:
    # NOTE: [:2] is added to avoid errors, remove later
    positions = x[np.argmin(sn, axis=1)]
    assert len(positions) > 0, "Positions array should contain elements"
    boundary = positions.max() - mesh_size
    idx = int(np.argmax(positions >= boundary))
    return positions[:idx], idx


# %% --- Front threshold definitions --------------------------------------------


def self_similar_front_threshold(
    run: file_io.RunData,
    theta_front: DimensionlessApertureAtFront,
    is_pressure: bool = True,
) -> FrontDetectionThreshold:
    """Self-similar threshold:
    pressure -  theta_front * sqrt(q0^2 * t / D) * k_n.
    aperture - w_i + theta_front * sqrt(q0^2 * t / D)."""
    D = diffusivity(run.params)
    q = run.params.q_0 or run.params.q
    assert q is not None and q > 0, f"expected correct injection rate, got {q}"
    scale = np.sqrt(q**2 * run.t / D)
    w_front = run.params.w_i + theta_front * scale
    p_front = theta_front * scale * run.params.k_n
    return p_front if is_pressure else w_front


def constant_aperture_threshold(
    run: file_io.RunData, pct_increase: FractionalPercentage
) -> FrontDetectionThreshold:
    return np.full_like(run.t, run.params.w_i * (1 + pct_increase))


def constant_pressure_threshold(
    t: Time, pc: CriticalPressure
) -> FrontDetectionThreshold:
    return np.full_like(t, pc)
