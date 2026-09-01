"""Power-law analysis of front and injection-point histories.

One :class:`Fit` type and one pair of entry points
(:func:`analyze_front`, :func:`analyze_p_inj`) cover both observables and
both aperture regimes (:data:`RIGID`, :data:`SOFT`).
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np

from fracinj import detection, io, physics
from fracinj.math_utils import fit_power_law
from fracinj.solvers.exact import solve_linear_diffusion_const_flux
from fracinj.solvers.similarity import solve_neumann_n3
from fracinj.types import Vector

Window = Literal["all", "early", "late"]


@dataclass(frozen=True)
class Regime:
    """A self-similar aperture regime and its power-law exponents."""

    name: str
    front_exponent: float
    p_inj_exponent: float


RIGID = Regime("rigid", front_exponent=0.5, p_inj_exponent=0.5)
SOFT = Regime("soft", front_exponent=0.8, p_inj_exponent=0.2)


@dataclass
class Fit:
    """A numerical series ``y(t)`` with its empirical (and optional analytical)
    power-law model ``A * t**alpha``.
    """

    quantity: Literal["front", "p_inj"]
    t: Vector
    y: Vector
    A_emp: float
    alpha_emp: float
    regime: Regime | None = None
    A_ana: float | None = None
    alpha_ana: float | None = None
    zeta: float | None = None

    def empirical(self, t: Vector | None = None) -> Vector:
        t = self.t if t is None else t
        return self.A_emp * t**self.alpha_emp

    def analytical(self, t: Vector | None = None) -> Vector:
        if self.A_ana is None:
            raise ValueError("no analytical model; pass a `regime` to analyze_*")
        t = self.t if t is None else t
        assert(self.alpha_ana is not None)
        return self.A_ana * t**self.alpha_ana


# --- series extraction --------------------------------------------------------


def _resolve_slice(run: io.RunData, window: Window, slc: slice | None) -> slice:
    if slc is not None:
        return slc
    if window == "all":
        return slice(None)
    return physics.time_slice(run.t, run.params, early=(window == "early"))


def _front_series(
    run: io.RunData, threshold: Vector | None, stress_front: bool, slc: slice
) -> tuple[Vector, Vector]:
    if stress_front:
        x_front, idx_cut = detection.find_stress_front(
            run.x_sc, run.sn[slc], L=run.params.L
        )
        return run.t[slc][:idx_cut], x_front
    if threshold is not None:
        x_front, has_crossing = detection.find_field_front(
            run.x_sc, run.p[slc], threshold
        )
        return run.t[slc][has_crossing], x_front
    raise ValueError("provide `threshold` or set stress_front=True")


def _p_inj_series(run: io.RunData, slc: slice) -> tuple[Vector, Vector]:
    _, p = io.sort_fields(run.x_sc, run.p)
    return run.t[slc], p[slc, 0]


# --- analytical amplitudes ---------------------------------------------------

# TODO: add analytical amplitudes for ramp injection rate

def _front_amplitude(regime: Regime, run: io.RunData, t: Vector, x: Vector) -> tuple[float, float]:
    """Return ``(A_ana, zeta)`` for the front position ``x_f = zeta * scale(t)``."""
    mask = t != 0
    if not np.any(mask):
        raise ValueError("no non-zero front times -- cannot estimate zeta")
    p = run.params
    if regime is RIGID:
        D = physics.diffusivity(p)
        scale = (D * t[mask]) ** regime.front_exponent
        amp_unit = D**regime.front_exponent
    elif regime is SOFT:
        a, q = physics.mobility(p), p.q_0
        scale = a ** (1 / 5) * q ** (3 / 5) * t[mask] ** regime.front_exponent
        amp_unit = (a * q**3) ** (1 / 5)
    else:  # pragma: no cover
        raise ValueError(f"unknown regime {regime!r}")
    zeta = float(np.mean(x[mask] / scale))
    return zeta * amp_unit, zeta


_RIGID_SIMILARITY = np.linspace(0.0, 10.0, 200)


def _p_inj_amplitude(regime: Regime, run: io.RunData) -> float:
    """Asymptotic prefactor of ``p_inj(t) ~ A * t**alpha``."""
    p = run.params
    if regime is RIGID:
        D = physics.diffusivity(p)
        if D <= 0:
            raise ValueError(f"diffusivity must be positive, got {D}")
        theta0 = solve_linear_diffusion_const_flux(_RIGID_SIMILARITY)[0]
        return theta0 * p.q_0 * p.k_n / D**regime.p_inj_exponent
    if regime is SOFT:
        w_final_inj = run.w[-1][0]
        if w_final_inj <= p.w_i:
            raise ValueError(
                f"final injection aperture ({w_final_inj}) <= w_i ({p.w_i}); "
                "run may not have reached the soft regime"
            )
        a = physics.mobility(p)
        if a <= 0 or p.q_0 == 0:
            raise ValueError(f"need a>0 and q_0!=0, got a={a}, q_0={p.q_0}")
        theta0 = solve_neumann_n3(p.w_i / w_final_inj)[1][0]
        return theta0 * p.k_n * (p.q_0**2 / a) ** regime.p_inj_exponent
    raise ValueError(f"unknown regime {regime!r}")  # pragma: no cover


# --- entry points -----------------------------------------------------------


def analyze_front(
    run: io.RunData,
    *,
    regime: Regime | None = None,
    window: Window = "all",
    threshold: Vector | None = None,
    stress_front: bool = False,
    slc: slice | None = None,
) -> Fit:
    """Fit the front position history; add an analytical model if ``regime`` is given."""
    slc = _resolve_slice(run, window, slc)
    t, x = _front_series(run, threshold, stress_front, slc)
    A_emp, alpha_emp = fit_power_law(t, x)
    fit = Fit("front", t=t, y=x, A_emp=A_emp, alpha_emp=alpha_emp, regime=regime)
    if regime is not None:
        fit.A_ana, fit.zeta = _front_amplitude(regime, run, t, x)
        fit.alpha_ana = regime.front_exponent
    return fit


def analyze_p_inj(
    run: io.RunData,
    *,
    regime: Regime | None = None,
    window: Window = "all",
    slc: slice | None = None,
) -> Fit:
    """Fit the injection-point pressure history; add an analytical model if ``regime`` is given."""
    slc = _resolve_slice(run, window, slc)
    t, p_inj = _p_inj_series(run, slc)
    A_emp, alpha_emp = fit_power_law(t, p_inj)
    fit = Fit("p_inj", t=t, y=p_inj, A_emp=A_emp, alpha_emp=alpha_emp, regime=regime)
    if regime is not None:
        fit.A_ana = _p_inj_amplitude(regime, run)
        fit.alpha_ana = regime.p_inj_exponent
    return fit


# --- named convenience wrappers -------------------------------------------


def front_rigid(run: io.RunData, **kw) -> Fit:
    return analyze_front(run, regime=RIGID, **kw)


def front_soft(run: io.RunData, **kw) -> Fit:
    return analyze_front(run, regime=SOFT, **kw)


def front_early(run: io.RunData, **kw) -> Fit:
    return analyze_front(run, regime=RIGID, window="early", **kw)


def front_late(run: io.RunData, **kw) -> Fit:
    return analyze_front(run, regime=SOFT, window="late", **kw)


def p_inj_early(run: io.RunData, **kw) -> Fit:
    return analyze_p_inj(run, regime=RIGID, window="early", **kw)


def p_inj_late(run: io.RunData, **kw) -> Fit:
    return analyze_p_inj(run, regime=SOFT, window="late", **kw)


# --- misc front diagnostics ------------------------------------------------


def local_exponent(t: Vector, x: Vector, window: int = 20) -> tuple[Vector, Vector]:
    """Rolling log-log slope of ``x(t)`` over a centred window of ``window`` points."""
    valid = (t > 0) & (x > 0)
    log_t, log_x = np.log(t[valid]), np.log(x[valid])
    n = log_t.size
    slope = np.full(n, np.nan)
    half = window // 2
    for i in range(half, n - half):
        s = slice(i - half, i + half)
        slope[i] = np.polyfit(log_t[s], log_x[s], deg=1)[0]
    return t[valid], slope


def mean_velocity(t: Vector, x: Vector) -> float:
    return (x[-1] - x[0]) / (t[-1] - t[0])
