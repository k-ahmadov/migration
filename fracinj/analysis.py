"""Power-law analysis of front and injection-point histories.

One :class:`Fit` type and one pair of entry points
(:func:`analyze_front`, :func:`analyze_p_inj`) cover both observables and
both aperture regimes (:data:`RIGID`, :data:`SOFT`). The named wrappers
(:func:`front_rigid`, :func:`p_inj_soft`, etc.) always pass a regime, so
they return :class:`AnalyticalFit` -- the same type with its analytical
fields narrowed from ``float | None`` to ``float``.
"""

from dataclasses import dataclass, fields
from typing import Callable, Literal

import numpy as np

from fracinj import detection, io, math_utils, physics
from fracinj.math_utils import compute_r_squared, fit_power_law
from fracinj.types import Parameters, Vector

Window = Literal["all", "early", "late"]


def check_out_rate(params: Parameters) -> float:
    q_0 = getattr(params, "q_0", 0.0) or 0.0
    m_q = getattr(params, "m_q", 0.0) or 0.0

    if q_0 > 0.0 and m_q > 0.0:
        raise ValueError(
            "both q_0 and m_q are set; the rate boundary condition is ambiguous"
        )
    if q_0 > 0.0:
        return q_0
    if m_q > 0.0:
        return m_q
    raise ValueError("no rate boundary condition: neither q_0 nor m_q is set")


@dataclass(frozen=True)
class Regime:
    """A self-similar aperture regime and its power-law exponents."""

    name: str
    n: float
    i: float
    coefficient: Callable[[Parameters], float]
    # TODO: make it be determined by the similarity ode solver
    theta0: float

    def _np2(self):
        return self.n + 2

    def front_exponent(self):
        return physics.P(self.n, self.i) / self._np2()

    def p_inj_exponent(self):
        return (1 + 2 * self.i) / self._np2()

    def front_amplitude(self, params: Parameters):
        a = self.coefficient(params)
        q = check_out_rate(params)
        return (a * q**self.n) ** (1 / self._np2())

    def p_inj_amplitude(self, params: Parameters):
        a = self.coefficient(params)
        q = check_out_rate(params)
        return (q**2 / a) ** (1 / self._np2())


RIGID_CONST = Regime(
    name="rigid",
    n=0,
    i=0,
    coefficient=physics.diffusivity,
    theta0=1.126,
)

SOFT_CONST = Regime(name="soft", n=3, i=0, coefficient=physics.mobility, theta0=1.328)


@dataclass(frozen=True)
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
        assert self.alpha_ana is not None
        return self.A_ana * t**self.alpha_ana


@dataclass(frozen=True)
class AnalyticalFit(Fit):
    """A :class:`Fit` whose analytical model is guaranteed present.

    Returned by the named ``*_rigid``/``*_soft``/``*_early``/``*_late``
    wrappers below, which always pass a concrete ``regime`` -- so callers
    get ``regime``/``A_ana``/``alpha_ana`` as plain, non-optional values
    instead of having to narrow ``Fit``'s ``X | None`` fields themselves.
    ``zeta`` stays optional: only front fits compute it, p_inj fits don't.
    """

    regime: Regime = RIGID_CONST
    A_ana: float = 0.0
    alpha_ana: float = 0.0


def _as_analytical(fit: Fit) -> AnalyticalFit:
    """Narrow a regime-fitted ``Fit`` to :class:`AnalyticalFit`."""
    assert fit.regime is not None
    assert fit.A_ana is not None
    assert fit.alpha_ana is not None
    return AnalyticalFit(**{f.name: getattr(fit, f.name) for f in fields(Fit)})


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
# TODO: add analytical amplitudes for constant pressure injection


def _front_amplitude(
    regime: Regime, p: Parameters, t: Vector, x: Vector
) -> tuple[float, float]:
    """Return ``(A_ana, zeta)`` for the front position ``x_f = zeta * scale(t)``."""
    mask = t != 0
    if not np.any(mask):
        raise ValueError("no non-zero front times -- cannot estimate zeta")
    amp_unit = regime.front_amplitude(p)
    scale = amp_unit * t[mask] ** regime.front_exponent()
    zeta = float(np.mean(x[mask] / scale))
    return zeta * amp_unit, zeta


def _p_inj_amplitude(regime: Regime, p: Parameters) -> float:
    """Asymptotic prefactor of ``p_inj(t) ~ A * t**alpha``."""
    return p.k_n * regime.theta0 * regime.p_inj_amplitude(p)


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
    if regime is None:
        return Fit("front", t=t, y=x, A_emp=A_emp, alpha_emp=alpha_emp)
    A_ana, zeta = _front_amplitude(regime, run.params, t, x)
    return Fit(
        "front",
        t=t,
        y=x,
        A_emp=A_emp,
        alpha_emp=alpha_emp,
        regime=regime,
        A_ana=A_ana,
        alpha_ana=regime.front_exponent(),
        zeta=zeta,
    )


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
    if regime is None:
        return Fit("p_inj", t=t, y=p_inj, A_emp=A_emp, alpha_emp=alpha_emp)
    A_ana = _p_inj_amplitude(regime, run.params)
    return Fit(
        "p_inj",
        t=t,
        y=p_inj,
        A_emp=A_emp,
        alpha_emp=alpha_emp,
        regime=regime,
        A_ana=A_ana,
        alpha_ana=regime.p_inj_exponent(),
    )


# --- named convenience wrappers -------------------------------------------


def front_rigid(run: io.RunData, **kw) -> AnalyticalFit:
    return _as_analytical(analyze_front(run, regime=RIGID_CONST, **kw))


def front_soft(run: io.RunData, **kw) -> AnalyticalFit:
    return _as_analytical(analyze_front(run, regime=SOFT_CONST, **kw))


def front_early(run: io.RunData, **kw) -> AnalyticalFit:
    return _as_analytical(analyze_front(run, regime=RIGID_CONST, window="early", **kw))


def front_late(run: io.RunData, **kw) -> AnalyticalFit:
    return _as_analytical(analyze_front(run, regime=SOFT_CONST, window="late", **kw))


def p_inj_rigid(run: io.RunData, **kw) -> AnalyticalFit:
    return _as_analytical(analyze_p_inj(run, regime=RIGID_CONST, **kw))


def p_inj_soft(run: io.RunData, **kw) -> AnalyticalFit:
    return _as_analytical(analyze_p_inj(run, regime=SOFT_CONST, **kw))


def p_inj_early(run: io.RunData, **kw) -> AnalyticalFit:
    return _as_analytical(analyze_p_inj(run, regime=RIGID_CONST, window="early", **kw))


def p_inj_late(run: io.RunData, **kw) -> AnalyticalFit:
    return _as_analytical(analyze_p_inj(run, regime=SOFT_CONST, window="late", **kw))


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


def analytical_fit_and_split(t_pct, d_pct, split_frac):
    """Split into early/late series and fit a fixed-exponent power law to each."""
    idx_start = int(len(t_pct) * split_frac)
    t_early, d_early = t_pct[:idx_start], d_pct[:idx_start]
    t_late, d_late = t_pct[idx_start:], d_pct[idx_start:]
    A_e = math_utils.fit_power_law_fixed_exponent(
        t_early, d_early, RIGID_CONST.front_exponent()
    )
    A_l = math_utils.fit_power_law_fixed_exponent(
        t_late, d_late, SOFT_CONST.front_exponent()
    )
    return (
        (
            t_early,
            A_e * t_early ** RIGID_CONST.front_exponent(),
            RIGID_CONST.front_exponent(),
            A_e,
        ),
        (
            t_late,
            A_l * t_late ** SOFT_CONST.front_exponent(),
            SOFT_CONST.front_exponent(),
            A_l,
        ),
    )


# --- field-data crossover fits -------------------------------------------


@dataclass(frozen=True)
class CrossoverModel:
    """Fixed exponents and transition sharpness of a two-regime crossover."""

    alpha_early: float = 1 / 2
    alpha_late: float = 7 / 5
    delta: float = 0.3


@dataclass(frozen=True)
class FieldCase:
    """One field seismicity dataset, its per-bin percentile envelope, and the
    best-fit crossover curve -- everything a plot panel needs, in one immutable
    value. Build it with :func:`fit_field_crossover`; every field is populated.
    """

    title: str
    t: Vector  # raw event times [s]
    d: Vector  # raw event distances [m]
    t_c: float  # crossover time scale [s]
    percentile: float
    bin_size: int
    model: CrossoverModel
    t_env: Vector  # percentile-envelope times [s]
    d_env: Vector  # percentile-envelope distances [m]
    t_fit: Vector  # crossover-curve abscissa [s]
    d_fit: Vector  # crossover-curve ordinate [m]
    A_early: float  # fitted early-time prefactor
    A_late: float  # fitted late-time prefactor
    perr: Vector  # 1-sigma errors on (A_early, A_late)
    r_squared: float  # R^2


def fit_field_crossover(
    t: Vector,
    d: Vector,
    *,
    title: str,
    t_c: float,
    model: CrossoverModel,
    eval_based_envelope: bool = True,
    percentile: float = 90.0,
    bin_size: int = 50,
    n_fit: int = 200,
) -> FieldCase:
    """Take the per-bin percentile envelope of ``d(t)`` and fit the crossover
    model to it, optimising only the two prefactors (exponents, ``t_c`` and
    ``delta`` are held fixed by ``model``).
    """
    if len(t) != len(d):
        raise ValueError(f"t and d differ in length: {len(t)} vs {len(d)}")
    if not 0.0 < percentile < 100.0:
        raise ValueError(f"percentile must be in (0, 100), got {percentile}")
    if t_c <= 0.0:
        raise ValueError(f"t_c must be positive, got {t_c}")

    t_env, d_env = detection.percentile_envelope(
        t, d, bin_size=bin_size, percentile=percentile
    )
    (A_early, A_late), perr = math_utils.fit_crossover_to_data(
        t_env,
        d_env,
        alpha_early=model.alpha_early,
        alpha_late=model.alpha_late,
        delta=model.delta,
        t_c=t_c,
        p0=[1.0, 1.0],
    )
    if eval_based_envelope:
        t_positive = np.asarray(t_env)[np.asarray(t_env) > 0.0]
    else:
        t_positive = np.asarray(t)[np.asarray(t) > 0.0]
    t_fit = np.geomspace(t_positive.min(), t_positive.max(), n_fit)
    d_fit = math_utils.crossover(
        x=t_fit,
        a=A_early,
        alpha=model.alpha_early,
        b=A_late,
        beta=model.alpha_late,
        x0=t_c,
        delta=model.delta,
    )
    r_sq = compute_r_squared(
        d_env,
        math_utils.crossover(
            x=t_env,
            a=A_early,
            alpha=model.alpha_early,
            b=A_late,
            beta=model.alpha_late,
            x0=t_c,
            delta=model.delta,
        ),
    )
    return FieldCase(
        title=title,
        t=np.asarray(t),
        d=np.asarray(d),
        t_c=t_c,
        percentile=percentile,
        bin_size=bin_size,
        model=model,
        t_env=t_env,
        d_env=d_env,
        t_fit=t_fit,
        d_fit=d_fit,
        A_early=float(A_early),
        A_late=float(A_late),
        perr=perr,
        r_squared=r_sq,
    )
