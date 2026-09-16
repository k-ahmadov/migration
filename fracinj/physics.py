"""Material coefficients, diffusivity, and (non)dimensionalisation.

Characteristic scales for the nonlinear diffusion equation

    a * d/dx ( w^n dw/dx ) = dw/dt

Linear diffusion is the n = 0 case with a = D, so one code path covers both.

All scalings follow from two relations:

    t_c = x_c^2 / (a * w_c^n)                     (equation balance)
    a^(1+i) * w_c^P = q * x_c^(1+2i)              (flux balance)

with P = n(1+i) + 1.

The scaling layer (``P``, ``Scales``, ``characteristic_scales`` and the three
branch functions) is pure: it knows nothing about ``Parameters``. The physics
layer below it maps a ``Parameters`` instance onto those scales.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, replace

import numpy as np

from fracinj.types import Field, Parameters, Vector

__all__ = [
    "SCALE_DISPATCH",
    "P",
    "Scales",
    "characteristic_scales",
    "cubic_scales",
]

CUBIC_N = 3.0


def P(n: float, i: float) -> float:
    """Exponent group P = n(1+i) + 1.

    Equals 1 for linear diffusion (n=0), 4+3i for the cubic case (n=3).
    """
    return n * (1.0 + i) + 1.0


@dataclass(frozen=True)
class Scales:
    """Characteristic length, time and aperture.

    ``sn_c`` is the characteristic normal stress. It is left as ``None`` by the
    pure scaling functions, which have no access to the elastic moduli, and is
    attached by :func:`with_stress_scale`. ``None`` rather than ``0.0`` so that
    an unset stress scale fails loudly instead of silently normalising by zero.
    """

    x_c: float
    t_c: float
    w_c: float
    sn_c: float | None = None

    def check_balance(self, *, a: float, n: float, rel_tol: float = 1e-9) -> bool:
        """Verify t_c = x_c^2 / (a w_c^n)."""
        return math.isclose(self.t_c, self.x_c**2 / (a * self.w_c**n), rel_tol=rel_tol)

    def check_flux(
        self, *, a: float, n: float, q: float, i: float, rel_tol: float = 1e-9
    ) -> bool:
        """Verify a w_c^(n+1) / x_c = q t_c^i."""
        return math.isclose(
            a * self.w_c ** (n + 1) / self.x_c, q * self.t_c**i, rel_tol=rel_tol
        )

    def require_stress(self) -> float:
        """Return ``sn_c``, raising if it was never attached."""
        if self.sn_c is None:
            raise ValueError(
                "stress scale not set; build these scales via cubic_scales() "
                "or pass them through with_stress_scale()"
            )
        return self.sn_c


def _validate(*, a: float, n: float) -> None:
    if a <= 0.0:
        raise ValueError(f"diffusivity a must be positive, got {a}")
    if n < 0.0:
        raise ValueError(f"exponent n must be non-negative, got {n}")


def constant_aperture(*, L: float, a: float, w_0: float, n: float) -> Scales:
    """Fixed aperture boundary condition w(0,t) = w_0.

    Both w_c and x_c are pinned by the problem statement.
    """
    _validate(a=a, n=n)
    if w_0 <= 0.0 or L <= 0.0:
        raise ValueError("w_0 and L must be positive")
    return Scales(x_c=L, t_c=L**2 / (a * w_0**n), w_c=w_0)


def prescribed_rate_fixed_aperture(
    *, a: float, w_i: float, q: float, i: float, n: float
) -> Scales:
    """Prescribed flux -a w^n w_x = q t^i, with the aperture pinned to w_c = w_i."""
    _validate(a=a, n=n)
    if w_i <= 0.0 or q <= 0.0:
        raise ValueError("w_i and q must be positive")
    if 1.0 + 2.0 * i <= 0.0:
        raise ValueError(
            f"this branch requires i > -1/2 (got i={i}); "
            "pin the length instead via pin='length'"
        )

    e = 1.0 / (1.0 + 2.0 * i)
    t_c = (a * w_i ** (n + 2) / q**2) ** e
    x_c = (a ** (1.0 + i) * w_i ** P(n, i) / q) ** e
    return Scales(x_c=x_c, t_c=t_c, w_c=w_i)


def prescribed_rate_fixed_length(
    *, L: float, a: float, q: float, i: float, n: float
) -> Scales:
    """Prescribed flux -a w^n w_x = q t^i, with the length pinned to x_c = L."""
    _validate(a=a, n=n)
    if L <= 0.0 or q <= 0.0:
        raise ValueError("L and q must be positive")

    p = P(n, i)
    if p <= 0.0:
        raise ValueError(f"P = n(1+i)+1 must be positive, got {p}")

    # Closed forms: valid at n = 0, unlike inverting t_c = L^2/(a w_c^n).
    t_c = (L ** (n + 2) / (a * q**n)) ** (1.0 / p)
    w_c = (L ** (1.0 + 2.0 * i) * q / a ** (1.0 + i)) ** (1.0 / p)
    return Scales(x_c=L, t_c=t_c, w_c=w_c)


SCALE_DISPATCH: dict[tuple[str, str], Callable[..., Scales]] = {
    ("aperture", "length"): constant_aperture,
    ("rate", "aperture"): prescribed_rate_fixed_aperture,
    ("rate", "length"): prescribed_rate_fixed_length,
}


def characteristic_scales(boundary: str, pin: str, **kwargs) -> Scales:
    """Dispatch to the right scaling.

    boundary : 'aperture' for w(0,t) = w_0, 'rate' for -a w^n w_x = q t^i
    pin      : which scale is held fixed, 'aperture' (w_c = w_i) or 'length' (x_c = L)

    Extra keyword arguments are forwarded to the selected function.
    """
    try:
        fn = SCALE_DISPATCH[(boundary, pin)]
    except KeyError:
        valid = ", ".join(f"{b}/{p}" for b, p in sorted(SCALE_DISPATCH))
        raise ValueError(
            f"no scaling for boundary={boundary!r}, pin={pin!r}; valid: {valid}"
        ) from None
    return fn(**kwargs)


# --------------------------------------------------------------------------
# Physics layer: Parameters -> Scales
# --------------------------------------------------------------------------


def _reference_aperture(params: Parameters) -> float:
    """Aperture the linear diffusivity is linearised about."""
    if params.DP > 0.0:
        return params.w_i + params.DP / params.k_n
    return params.w_i


def mobility(params: Parameters) -> float:
    """Transmissivity coefficient ``k_n / (12 mu)`` (the ``a`` of Murphy 2004)."""
    return params.k_n / (12 * params.mu)


def diffusivity(params: Parameters) -> float:
    """Linear hydraulic diffusivity ``mobility * w_ref**3``."""
    return mobility(params) * _reference_aperture(params) ** 3


def plane_strain_modulus(params: Parameters) -> float:
    return params.E / (1 - params.nu**2)


def characteristic_stress(w_c: float, params: Parameters) -> float:
    """Normal stress scale ``w_c * E' / L``."""
    return w_c * plane_strain_modulus(params) / params.L


def with_stress_scale(scales: Scales, params: Parameters) -> Scales:
    """Return a copy of ``scales`` with ``sn_c`` filled in."""
    return replace(scales, sn_c=characteristic_stress(scales.w_c, params))


def _rate_bc(params: Parameters) -> tuple[float, float]:
    """Return ``(q, i)`` for whichever rate boundary condition is configured.

    ``q`` and ``i`` are two halves of one boundary condition, so they are always
    derived together: a constant rate is (q_0, 0), a ramp is (m_q, 1). Choosing
    them independently silently mismatches the units of ``q`` with the exponent.
    """
    q_0 = getattr(params, "q_0", 0.0) or 0.0
    m_q = getattr(params, "m_q", 0.0) or 0.0

    if q_0 > 0.0 and m_q > 0.0:
        raise ValueError(
            "both q_0 and m_q are set; the rate boundary condition is ambiguous"
        )
    if q_0 > 0.0:
        return q_0, 0.0
    if m_q > 0.0:
        return m_q, 1.0
    raise ValueError("no rate boundary condition: neither q_0 nor m_q is set")

def is_constant_rate(params: Parameters) -> bool:
    _, i = _rate_bc(params)
    return i == 0.0



def cubic_scales(
    params: Parameters, *, pin: str = "aperture", ramp: bool | None = None
) -> Scales:
    """Characteristic scales for the cubic (n=3) prescribed-rate problem.

    pin  : 'aperture' pins w_c = w_i, 'length' pins x_c = L.
    ramp : force the ramp (i=1) or constant (i=0) branch. ``None`` infers it
           from whichever of ``q_0`` / ``m_q`` is set.
    """
    if ramp is None:
        q, i = _rate_bc(params)
    elif ramp:
        q, i = params.m_q, 1.0
    else:
        q, i = params.q_0, 0.0

    if pin == "aperture":
        pinned = {"w_i": params.w_i}
    elif pin == "length":
        pinned = {"L": params.L}
    else:
        raise ValueError(f"pin must be 'aperture' or 'length', got {pin!r}")

    scales = characteristic_scales(
        "rate", pin, a=mobility(params), q=q, i=i, n=CUBIC_N, **pinned
    )
    return with_stress_scale(scales, params)


# Thin aliases, kept for existing call sites.
def const_rate_fixed_aperture_cubic(p: Parameters) -> Scales:
    return cubic_scales(p, pin="aperture", ramp=False)


def ramp_rate_fixed_aperture_cubic(p: Parameters) -> Scales:
    return cubic_scales(p, pin="aperture", ramp=True)


def const_rate_fixed_length_cubic(p: Parameters) -> Scales:
    return cubic_scales(p, pin="length", ramp=False)


def ramp_rate_fixed_length_cubic(p: Parameters) -> Scales:
    return cubic_scales(p, pin="length", ramp=True)


# --------------------------------------------------------------------------
# Time-series helpers
# --------------------------------------------------------------------------


def time_slice(t: Vector, p: Parameters, *, early: bool) -> slice:
    """Slice selecting the early- or late-time portion of ``t``.

    The split is at ``t_c`` for the aperture-pinned cubic scaling. The rate
    exponent is no longer an argument: it is fixed by the boundary condition
    configured on ``p``.
    """
    t_c = cubic_scales(p, pin="aperture").t_c
    idx = min(int(np.searchsorted(t, t_c)) + 1, len(t))
    return slice(idx) if early else slice(idx, None)


def _time_column(t: Vector | float, w: Field | Vector) -> np.ndarray:
    """Reshape ``t`` so it broadcasts against ``w``.

    Three accepted pairings:
      * scalar ``t`` with a single 1D profile ``w``
      * 1D ``t`` with 1D ``w`` of the same length (elementwise, one point per time)
      * 1D ``t`` of length n_t with a ``(n_t, n_x)`` stack ``w``
    """
    t_ = np.asarray(t, dtype=float)
    w_ = np.asarray(w)

    if t_.ndim == 0:
        if w_.ndim != 1:
            raise ValueError(f"w must be 1D when t is scalar, got shape {w_.shape}")
        return t_

    if t_.ndim != 1:
        raise ValueError(f"t must be scalar or 1D, got shape {t_.shape}")

    if w_.ndim == 1:
        # Elementwise pairing: one aperture value per time.
        if t_.shape != w_.shape:
            raise ValueError(
                f"1D t and 1D w must match elementwise, got {t_.shape} and {w_.shape}"
            )
        return t_

    if w_.ndim == 2:
        if t_.shape[0] != w_.shape[0]:
            raise ValueError(
                f"t has {t_.shape[0]} entries but w has {w_.shape[0]} rows"
            )
        return t_[:, None]

    raise ValueError(f"w must be 1D or 2D, got shape {w_.shape}")
