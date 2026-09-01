"""Material coefficients, diffusivity, and (non)dimensionalisation."""

from typing import Literal, NamedTuple

import numpy as np

from fracinj.types import Field, Parameters, Vector

RateBC = Literal["const_rate", "ramp_rate"]


class Scales(NamedTuple):
    """Characteristic aperture and time used to (non)dimensionalise a run."""

    w: float
    t: float
    sn: float


def mobility(params: Parameters) -> float:
    """Transmissivity coefficient ``k_n / (12 mu)`` (the ``a`` of Murphy 2004)."""
    return params.k_n / (12 * params.mu)


def _reference_aperture(params: Parameters) -> float:
    """Aperture the linear diffusivity is linearised about."""
    if params.DP != 0.0:
        return params.w_i + params.DP / params.k_n
    return params.w_i


def diffusivity(params: Parameters) -> float:
    """Linear hydraulic diffusivity ``mobility * w_ref**3``."""
    return mobility(params) * _reference_aperture(params) ** 3


def plane_strain_modulus(params: Parameters) -> float:
    return params.E / (1 - params.nu**2)


def dimensionalize(params: Parameters, bc: RateBC = "const_rate") -> Scales:
    """Characteristic aperture/time scales for the chosen left boundary condition."""
    a = mobility(params)
    if bc == "const_rate":
        if params.q_0 == 0.0:
            raise ValueError("dimensionalize(bc='const_rate') needs params.q_0 != 0")
        w_char = (params.L * params.q_0 / a) ** 0.25
    elif bc == "ramp_rate":
        if params.m_q == 0.0:
            raise ValueError("dimensionalize(bc='ramp_rate') needs params.m_q != 0")
        w_char = (params.L**3 * params.m_q / a**2) ** (1 / 7)
    else:  # pragma: no cover - guarded by the Literal type
        raise ValueError(f"unknown boundary condition {bc!r}")
    sn_char = w_char * plane_strain_modulus(params) / params.L
    t_char = params.L**2 / (a * w_char**3)
    return Scales(w=w_char, t=t_char, sn=sn_char)


# TODO: add ramp injection rate to time slicing 
def time_slice(t: Vector, params: Parameters, *, early: bool) -> slice:
    """Slice selecting the early- or late-time portion of ``t``."""
    idx = int(np.searchsorted(t, dimensionalize(params).t)) + 1
    return slice(idx) if early else slice(idx, None)


def _time_column(t: Vector | float, w: Field | Vector) -> np.ndarray:
    """Reshape ``t`` so it broadcasts against ``w``.

    ``w`` may be a single profile (1D) paired with a scalar ``t``, or a
    ``(n_t, n_x)`` stack paired with a ``(n_t,)`` time vector.
    """
    t_ = np.asarray(t, dtype=float)
    if t_.ndim == 0 or np.ndim(w) == 1:
        assert np.ndim(w) == 1, "w must be 1D when t is scalar"
        return t_
    return t_[:, None]


def nondimensionalize_rigid(
    *, x: Vector, t: Vector | float, w: Field | Vector, params: Parameters
) -> tuple[np.ndarray, np.ndarray]:
    """Similarity variables for the rigid (linear-diffusion) regime."""
    D = diffusivity(params)
    tc = _time_column(t, w)
    zeta = x / np.sqrt(D * tc)
    theta = (w - params.w_i) / (params.q_0 * np.sqrt(tc / D))
    return zeta, theta


def nondimensionalize_soft(
    *, x: Vector, t: Vector | float, w: Field | Vector, params: Parameters
) -> tuple[np.ndarray, np.ndarray]:
    """Similarity variables for the soft (n=3 nonlinear-diffusion) regime."""
    a = mobility(params)
    q = params.q_0
    tc = _time_column(t, w)
    zeta = x / (a * q**3 * tc**4) ** (1 / 5)
    theta = w / (q**2 * tc / a) ** (1 / 5)
    return zeta, theta
