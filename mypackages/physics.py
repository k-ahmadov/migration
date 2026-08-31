from functools import partial

import numpy as np
from scipy.optimize import curve_fit

from mypackages import file_io
from mypackages.typesdefs import (
    Field,
    Float64,
    OneD,
    Parameters,
    TwoD,
    Vector,
)


def parameter_a(params: Parameters) -> float:
    return params.k_n / (12 * params.mu)


def diffusivity(params: Parameters) -> float:
    if params.DP != 0.0:
        w_0 = params.w_i + params.DP / params.k_n
        return parameter_a(params) * w_0**3
    return parameter_a(params) * params.w_i**3


def power_law(t, A, alpha):
    return A * t**alpha


def fit_front_power_law(
    t: Vector,
    x_front: Vector,
) -> tuple[float, float]:
    (A, alpha), _ = curve_fit(
        power_law,
        t,
        x_front,
        p0=(x_front.max(), 0.8),
        maxfev=20_000,
    )
    return A, alpha


def fit_front_power_law_fixed_alpha(
    t: Vector,
    x_front: Vector,
    alpha: float,
) -> float:
    power_law_fixed_exponent = partial(power_law, alpha=alpha)
    (A,), _ = curve_fit(
        power_law_fixed_exponent,
        t,
        x_front,
        p0=(x_front.max(),),
    )
    return A


def critical_time(params: Parameters) -> float:
    t_c = parameter_a(params) * params.w_i**5 / params.q_0**2
    return t_c


def time_slice(run: file_io.RunData, early: bool) -> slice:
    t_c = critical_time(run.params)
    idx = np.searchsorted(run.t, t_c) + 1
    return slice(idx) if early else slice(idx, None)


def nondimensionalize_rigid(
    *,
    x: Vector,
    t: Vector,
    w: Field | np.ndarray[OneD, Float64],
    params: Parameters,
) -> tuple[np.ndarray[TwoD | OneD, Float64], np.ndarray[TwoD | OneD, Float64]]:
    D = diffusivity(params)
    wi = params.w_i
    q = params.q_0
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
    x: Vector,
    t: Vector | float,
    w: Field,
    params: Parameters,
) -> tuple[np.ndarray, np.ndarray]:
    M = parameter_a(params)
    q = params.q_0
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


def dimensionalize(params: Parameters, left_bc_constant_rate: bool = True):
    if left_bc_constant_rate:
        w_char = (params.L * params.q_0 / parameter_a(params)) ** 0.25
    else:
        w_char = (params.L**3 * params.m_q / parameter_a(params)**2)**(1/7)
    t_char = (params.L ** 2) / (parameter_a(params) * (w_char**3))
    return w_char, t_char
