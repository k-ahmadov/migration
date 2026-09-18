"""Power-law analysis of front and injection-point histories.

One :class:`Fit` type and one pair of entry points (:func:`analyze_front`,
:func:`analyze_p_inj`) cover both observables, both aperture regimes and both
injection rates.

Exponents are derived from ``(n, i)`` rather than stored:

    front = P(n, i) / (n + 2)        P = n(1+i) + 1
    p_inj = (1 + 2i) / (n + 2)

    regime         n  i   front   p_inj
    rigid/const    0  0   1/2     1/2
    soft /const    3  0   4/5     1/5
    rigid/ramp     0  1   1/2     3/2
    soft /ramp     3  1   7/5     3/5

The rigid front exponent is 1/2 for both rates: a linear diffusion front goes
as sqrt(D t) however the fluid is injected, and only the injection-point
pressure feels the rate. soft/ramp gives 7/5, which is the exponent used as
``CrossoverModel.alpha_late``.
"""

from collections.abc import Callable
from dataclasses import dataclass, fields
from typing import Literal

import numpy as np

from fracinj import detection, io, math_utils, physics
from fracinj.math_utils import fit_power_law
from fracinj.types import Parameters, Vector

Quantity = Literal["front", "p_inj"]
RegimeName = Literal["rigid", "soft"]
Rate = Literal["const", "ramp"]
Window = Literal["all", "early", "late"]

# theta0 for linear diffusion under constant flux: w(0,t) = 2q sqrt(t/D)/sqrt(pi)
# against the scale (q^2 t / D)^(1/2).
THETA0_RIGID_CONST = 2.0 / np.sqrt(np.pi)  # 1.128379...
THETA0_SOFT_CONST = 1.328


def rate_of(params: Parameters, expect_i: float | None = None) -> float:
    """Return ``q`` for whichever rate boundary condition is configured.

    ``expect_i`` guards against analysing a run with a mismatched regime: a
    ramp run read through a constant-rate regime would otherwise feed ``m_q``
    into ``i=0`` formulas and return a plausible, wrong number.
    """
    q, i = physics._rate_bc(params)
    if expect_i is not None and i != expect_i:
        raise ValueError(
            f"regime expects i={expect_i:g}, but the run is configured with "
            f"i={i:g}; pass the matching regime"
        )
    return q


# --- regime ------------------------------------------------------------------


@dataclass(frozen=True)
class Regime:
    """A self-similar aperture regime, parameterised by ``(n, i)``."""

    name: RegimeName
    rate: Rate
    n: float
    i: float
    coefficient: Callable[[Parameters], float]
    # TODO: determine theta0 from the similarity ODE solver.
    # None means no analytical p_inj amplitude is available for this regime.
    theta0: float | None

    @property
    def _np2(self) -> float:
        return self.n + 2.0

    @property
    def distance_exponent(self) -> float:
        return physics.P(self.n, self.i) / self._np2

    @property
    def aperture_exponent(self) -> float:
        return (1.0 + 2.0 * self.i) / self._np2

    def _coefficient(self, params: Parameters) -> float:
        a = self.coefficient(params)
        if a <= 0.0:
            raise ValueError(f"coefficient must be positive, got {a}")
        return a

    def distance_amplitude(self, params: Parameters) -> float:
        """Similarity length amplitude ``(a q^n)^(1/(n+2))``."""
        a = self._coefficient(params)
        return (a * rate_of(params, self.i) ** self.n) ** (1.0 / self._np2)

    def aperture_amplitude(self, params: Parameters) -> float:
        """Similarity aperture amplitude ``(q^2/a)^(1/(n+2))``."""
        a = self._coefficient(params)
        return (rate_of(params, self.i) ** 2 / a) ** (1.0 / self._np2)


REGIME_DISPATCH: dict[tuple[RegimeName, Rate], Regime] = {
    ("rigid", "const"): Regime(
        name="rigid",
        rate="const",
        n=0.0,
        i=0.0,
        coefficient=physics.diffusivity,
        theta0=THETA0_RIGID_CONST,
    ),
    ("soft", "const"): Regime(
        name="soft",
        rate="const",
        n=3.0,
        i=0.0,
        coefficient=physics.mobility,
        theta0=THETA0_SOFT_CONST,
    ),
    # Ramp exponents are exact; the amplitudes are not yet derived, so theta0
    # is None. Front fits still work -- zeta is estimated from the data.
    ("rigid", "ramp"): Regime(
        name="rigid",
        rate="ramp",
        n=0.0,
        i=1.0,
        coefficient=physics.diffusivity,
        theta0=THETA0_RIGID_CONST,
    ),
    ("soft", "ramp"): Regime(
        name="soft",
        rate="ramp",
        n=3.0,
        i=1.0,
        coefficient=physics.mobility,
        theta0=THETA0_SOFT_CONST,
    ),
}


def regime_for(name: RegimeName, rate: Rate = "const") -> Regime:
    """Look up a regime, listing valid keys on failure."""
    try:
        return REGIME_DISPATCH[(name, rate)]
    except KeyError:
        valid = ", ".join(f"{n}/{r}" for n, r in sorted(REGIME_DISPATCH))
        raise ValueError(f"no regime {name!r}/{rate!r}; valid: {valid}") from None


# --- fit types ---------------------------------------------------------------


@dataclass(frozen=True)
class Fit:
    """A numerical series ``y(t)`` with its empirical power-law model
    ``A_emp * t**alpha_emp``.

    Carries no analytical model. A fit that has one is an
    :class:`AnalyticalFit`, which *adds* the three fields rather than
    overriding optional versions of them -- so there is no ``X | None`` for a
    caller to narrow, and no field-override for a type checker to object to.
    """

    quantity: Quantity
    t: Vector
    y: Vector
    A_emp: float
    alpha_emp: float

    def empirical(self, t: Vector | None = None) -> Vector:
        t = self.t if t is None else t
        return self.A_emp * t**self.alpha_emp


@dataclass(frozen=True)
class AnalyticalFit(Fit):
    """A :class:`Fit` that also carries the analytical model for a regime.

    ``zeta`` stays optional: only front fits estimate it, p_inj fits don't.
    """

    regime: Regime
    A_ana: float
    alpha_ana: float
    zeta: float | None = None

    def analytical(self, t: Vector | None = None) -> Vector:
        t = self.t if t is None else t
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
        # p lives on the FVM grid (x_vert, Nx_p), not the elastic grid (x_sc).
        x_front, has_crossing = detection.find_field_front(
            run.x_vert, run.p[slc], threshold
        )
        return run.t[slc][has_crossing], x_front
    raise ValueError("provide `threshold` or set stress_front=True")


def _p_inj_series(run: io.RunData, slc: slice) -> tuple[Vector, Vector]:
    _, p = io.sort_fields(run.x_vert, run.p)
    return run.t[slc], p[slc, 0]


# --- analytical amplitudes ---------------------------------------------------


def _front_amplitude(
    regime: Regime, p: Parameters, t: Vector, x: Vector
) -> tuple[float, float]:
    """Return ``(A_ana, zeta)`` for the front position ``x_f = zeta * scale(t)``."""
    mask = t != 0
    if not np.any(mask):
        raise ValueError("no non-zero front times -- cannot estimate zeta")
    amp_unit = regime.distance_amplitude(p)
    scale = amp_unit * t[mask] ** regime.distance_exponent
    zeta = float(np.mean(x[mask] / scale))
    return zeta * amp_unit, zeta


def _p_inj_amplitude(regime: Regime, p: Parameters) -> float:
    """Asymptotic prefactor of ``p_inj(t) ~ A * t**alpha``."""
    if regime.theta0 is None:
        raise NotImplementedError(
            f"no analytical p_inj amplitude for the {regime.name}/{regime.rate} "
            "regime yet; the exponent is exact, so fit the prefactor "
            "empirically via analyze_p_inj(run) with regime=None"
        )
    return p.k_n * regime.theta0 * regime.aperture_amplitude(p)


# --- entry points -----------------------------------------------------------


def _empirical_fields(fit: Fit) -> dict:
    """The empirical half of a fit, as kwargs for :class:`AnalyticalFit`."""
    return {f.name: getattr(fit, f.name) for f in fields(Fit)}


def analyze_front(
    run: io.RunData,
    *,
    window: Window = "all",
    threshold: Vector | None = None,
    stress_front: bool = False,
    slc: slice | None = None,
) -> Fit:
    """Fit the front position history, empirically.

    For the analytical model of a given regime, see :func:`front_regime`.
    """
    slc = _resolve_slice(run, window, slc)
    t, x = _front_series(run, threshold, stress_front, slc)
    A_emp, alpha_emp = fit_power_law(t, x)
    return Fit("front", t=t, y=x, A_emp=A_emp, alpha_emp=alpha_emp)


def front_regime(run: io.RunData, regime: Regime, **kw) -> AnalyticalFit:
    """Fit the front position history and attach ``regime``'s analytical model.

    Extra keyword arguments go to :func:`analyze_front`.
    """
    fit = analyze_front(run, **kw)
    A_ana, zeta = _front_amplitude(regime, run.params, fit.t, fit.y)
    return AnalyticalFit(
        **_empirical_fields(fit),
        regime=regime,
        A_ana=A_ana,
        alpha_ana=regime.distance_exponent,
        zeta=zeta,
    )


def analyze_p_inj(
    run: io.RunData,
    *,
    window: Window = "all",
    slc: slice | None = None,
) -> Fit:
    """Fit the injection-point pressure history, empirically.

    For the analytical model of a given regime, see :func:`p_inj_regime`.
    """
    slc = _resolve_slice(run, window, slc)
    t, p_inj = _p_inj_series(run, slc)
    A_emp, alpha_emp = fit_power_law(t, p_inj)
    return Fit("p_inj", t=t, y=p_inj, A_emp=A_emp, alpha_emp=alpha_emp)


def p_inj_regime(run: io.RunData, regime: Regime, **kw) -> AnalyticalFit:
    """Fit the injection-point pressure and attach ``regime``'s analytical model.

    Extra keyword arguments go to :func:`analyze_p_inj`. Raises
    ``NotImplementedError`` for regimes whose ``theta0`` is not yet derived
    (currently both ramp regimes) -- use :func:`analyze_p_inj` for those.
    """
    fit = analyze_p_inj(run, **kw)
    return AnalyticalFit(
        **_empirical_fields(fit),
        regime=regime,
        A_ana=_p_inj_amplitude(regime, run.params),
        alpha_ana=regime.aperture_exponent,
    )


# --- analysis dispatch -------------------------------------------------------

# Each regime has a natural window: rigid dominates early, soft dominates late.
_NATURAL_WINDOW: dict[RegimeName, Window] = {"rigid": "early", "soft": "late"}

_ENTRY_POINTS: dict[Quantity, Callable[..., AnalyticalFit]] = {
    "front": front_regime,
    "p_inj": p_inj_regime,
}


def _wrap(
    fn: Callable[..., AnalyticalFit], regime: Regime, **fixed
) -> Callable[..., AnalyticalFit]:
    def wrapper(run: io.RunData, **kw) -> AnalyticalFit:
        return fn(run, regime, **fixed, **kw)

    return wrapper


def _build_analysis_dispatch() -> dict[
    tuple[Quantity, RegimeName, Rate, Window], Callable[..., AnalyticalFit]
]:
    """One wrapper per (quantity, regime, rate, window).

    Windows are restricted to "all" and each regime's natural window (early
    for rigid, late for soft) -- see the ``analyze`` docstring for why.
    """
    table: dict[tuple[Quantity, RegimeName, Rate, Window], Callable] = {}
    for quantity, analyze_fn in _ENTRY_POINTS.items():
        for (name, rate), regime in REGIME_DISPATCH.items():
            for window in ("all", _NATURAL_WINDOW[name]):
                key = (quantity, name, rate, window)
                table[key] = _wrap(analyze_fn, regime, window=window)
    return table


ANALYSIS_DISPATCH = _build_analysis_dispatch()


def analyze(
    run: io.RunData,
    quantity: Quantity,
    regime: RegimeName,
    rate: Rate = "const",
    window: Window = "all",
    **kw,
) -> AnalyticalFit:
    """Fit ``quantity`` for a given regime, rate and window.

    ``analyze(run, "front", "soft", window="late")`` replaces the old
    ``front_late_const(run)``; pass ``rate="ramp"`` for the ramp variants.
    """
    key = (quantity, regime, rate, window)
    try:
        fn = ANALYSIS_DISPATCH[key]
    except KeyError:
        natural = _NATURAL_WINDOW.get(regime)
        if natural is not None and window not in ("all", natural):
            raise ValueError(
                f"window {window!r} is not paired with the {regime!r} regime; "
                f"use 'all' or {natural!r}, or call analyze_{quantity} directly "
                "with an explicit slice"
            ) from None
        valid = ", ".join("/".join(k) for k in sorted(ANALYSIS_DISPATCH))
        raise ValueError(f"no analysis for {'/'.join(key)}; valid: {valid}") from None
    return fn(run, **kw)


# --- named wrappers, kept for existing call sites ---------------------------

front_rigid_const = ANALYSIS_DISPATCH["front", "rigid", "const", "all"]
front_early_const = ANALYSIS_DISPATCH["front", "rigid", "const", "early"]
front_soft_const = ANALYSIS_DISPATCH["front", "soft", "const", "all"]
front_late_const = ANALYSIS_DISPATCH["front", "soft", "const", "late"]
p_inj_rigid_const = ANALYSIS_DISPATCH["p_inj", "rigid", "const", "all"]
p_inj_early_const = ANALYSIS_DISPATCH["p_inj", "rigid", "const", "early"]
p_inj_soft_const = ANALYSIS_DISPATCH["p_inj", "soft", "const", "all"]
p_inj_late_const = ANALYSIS_DISPATCH["p_inj", "soft", "const", "late"]

front_rigid_ramp = ANALYSIS_DISPATCH["front", "rigid", "ramp", "all"]
front_early_ramp = ANALYSIS_DISPATCH["front", "rigid", "ramp", "early"]
front_soft_ramp = ANALYSIS_DISPATCH["front", "soft", "ramp", "all"]
front_late_ramp = ANALYSIS_DISPATCH["front", "soft", "ramp", "late"]
p_inj_rigid_ramp = ANALYSIS_DISPATCH["p_inj", "rigid", "ramp", "all"]
p_inj_early_ramp = ANALYSIS_DISPATCH["p_inj", "rigid", "ramp", "early"]
p_inj_soft_ramp = ANALYSIS_DISPATCH["p_inj", "soft", "ramp", "all"]
p_inj_late_ramp = ANALYSIS_DISPATCH["p_inj", "soft", "ramp", "late"]

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
        t_early, d_early, REGIME_DISPATCH["rigid", "const"].distance_exponent
    )
    A_l = math_utils.fit_power_law_fixed_exponent(
        t_late, d_late, REGIME_DISPATCH["soft", "const"].distance_exponent
    )
    return (
        (
            t_early,
            A_e * t_early ** REGIME_DISPATCH["rigid", "const"].distance_exponent,
            REGIME_DISPATCH["rigid", "const"].distance_exponent,
            A_e,
        ),
        (
            t_late,
            A_l * t_late ** REGIME_DISPATCH["soft", "const"].distance_exponent,
            REGIME_DISPATCH["soft", "const"].distance_exponent,
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
    r_sq = math_utils.compute_r_squared(
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
