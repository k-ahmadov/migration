from functools import partial

import numpy as np
from scipy.optimize import curve_fit

from mypackages.types import (
    CharacteristicTime,
    Field,
    Float64,
    HydraulicDiffusivity,
    OneD,
    ParameterM,
    Parameters,
    Prefactor,
    ScalingExponent,
    Time,
    TwoD,
    XPositions,
)


def parameter_a(params: Parameters) -> ParameterM:
    return params.k_n / (12 * params.mu)


def diffusivity(params: Parameters) -> HydraulicDiffusivity:
    DP = params.DP
    if DP is not None:
        w_0 = params.w_i + DP / params.k_n
        return parameter_a(params) * w_0**3
    return parameter_a(params) * params.w_i**3


def power_law(t, A, alpha):
    return A * t**alpha


def fit_front_power_law(
    t: Time,
    x_front: XPositions,
) -> tuple[Prefactor, ScalingExponent]:
    (A, alpha), _ = curve_fit(
        power_law,
        t,
        x_front,
        p0=(x_front.max(), 0.8),
        maxfev=20_000,
    )
    return A, alpha


def fit_front_power_law_fixed_alpha(
    t: Time,
    x_front: XPositions,
    alpha: ScalingExponent,
) -> Prefactor:
    power_law_fixed_exponent = partial(power_law, alpha=alpha)
    (A,), _ = curve_fit(
        power_law_fixed_exponent,
        t,
        x_front,
        p0=(x_front.max(),),
    )
    return A


def critical_time(params: Parameters) -> CharacteristicTime:
    q = params.q_0 or params.q
    assert q is not None
    t_c = parameter_a(params) * params.w_i**5 / q**2
    return t_c


def nondimensionalize_rigid(
    *,
    x: XPositions,
    t: Time,
    w: Field | np.ndarray[OneD, Float64],
    params: Parameters,
) -> tuple[np.ndarray[TwoD | OneD, Float64], np.ndarray[TwoD | OneD, Float64]]:
    D = diffusivity(params)
    wi = params.w_i
    q = params.q_0 or params.q
    assert q is not None
    t_ = np.asarray(t)

    if t_.ndim == 0 or w.ndim == 1:
        assert np.asarray(w).ndim == 1, "w must be 1D when t is scalar"
        t_col = t_
    else:
        t_col = t_[:, None]
    zeta = x / np.sqrt(D * t_col)
    theta = (w - wi) / (q * np.sqrt(t_col / D))
    return zeta, theta


def nondimensionalize_soft(
    *,
    x: XPositions,
    t: Time | float,
    w: Field,
    params: Parameters,
) -> tuple[np.ndarray, np.ndarray]:
    M = parameter_a(params)
    q = params.q_0 or params.q
    assert q is not None
    t_ = np.asarray(t)

    if t_.ndim == 0 or w.ndim == 1:
        assert np.asarray(w).ndim == 1, "w must be 1D when t is scalar"
        t_col = t_
    else:
        t_col = t_[:, None]

    zeta = x / (M * q**3 * t_col**4) ** (1 / 5)
    theta = w / (q**2 * t_col / M) ** (1 / 5)
    return zeta, theta


def dimensionalize(params: Parameters) -> tuple[float, float]:
    """Return characteristic aperture (w_char) and time (t_char)."""
    if params.L <= 0 or params.mu <= 0:
        raise ValueError("L and mu must be positive.")
    w_char = (params.L * params.flux / parameter_a(params)) ** 0.25
    t_char = (params.L * params.L) / (parameter_a(params) * (w_char**3))
    return w_char, t_char
