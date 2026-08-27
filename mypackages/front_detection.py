import numpy as np

from mypackages import file_io, physics
from mypackages.typesdefs import Bool, Field, OneD, Vector


def find_field_front(
    x: Vector,
    field: Field,
    threshold: Vector,
    interpolate: bool = True,
) -> tuple[Vector, np.ndarray[OneD, Bool]]:
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


def find_stress_front(x: Vector, sn: Field, L: float) -> tuple[Vector, int]:
    """Track the stress front position over time and find when it reaches
    within one mesh cell of the domain boundary L.

    The front at each timestep is taken as the x-position of minimum
    (most tensile) stress in that timestep's profile.

    Returns:
        positions: front x-position at each timestep, up to (not including)
            the timestep where the front first reaches `boundary`.
        idx: index of that first timestep, or len(positions_full) if the
            front never reaches the boundary within the given data.
    """
    if sn.shape[0] == 0:
        raise ValueError("sn must contain at least one timestep")

    positions = x[np.argmin(sn, axis=1)]
    mesh_size = x[-1] - x[-2]
    boundary = L - mesh_size

    reached = np.flatnonzero(positions >= boundary)  # True values are non-zero
    idx = int(reached[0]) if reached.size > 0 else len(positions)

    return positions[:idx], idx


# %% --- Front threshold definitions --------------------------------------------


def self_similar_front_threshold(
    run: file_io.RunData,
    theta_front: float,
    is_pressure: bool = True,
) -> Vector:
    """Self-similar threshold:
    pressure -  theta_front * sqrt(q0^2 * t / D) * k_n.
    aperture - w_i + theta_front * sqrt(q0^2 * t / D)."""
    D = physics.diffusivity(run.params)
    scale = np.sqrt(run.params.q_0**2 * run.t / D)
    w_front = run.params.w_i + theta_front * scale
    p_front = theta_front * scale * run.params.k_n
    return p_front if is_pressure else w_front


def constant_aperture_threshold(
    run: file_io.RunData, pct_increase: float
) -> Vector:
    return np.full_like(run.t, run.params.w_i * (1 + pct_increase))


def constant_pressure_threshold(
    t: Vector, pc: float
) -> Vector:
    return np.full_like(t, pc)
