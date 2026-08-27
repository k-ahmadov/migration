from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from mypackages import file_io, front_analysis, physics
from mypackages.typesdefs import Float64, OneD, Vector
from mysolvers.exact_solutions import solve_linear_diffusion_const_flux
from mysolvers.similarity_solutions import solve_neumann_n3


def find_p_inj(run: file_io.RunData) -> tuple[Vector, Vector]:
    """Injection-point pressure history, sorted by x_sc."""
    _, p = file_io.sort_fields(run.x_sc, run.p)
    return run.t, p[:, 0]


@dataclass
class PInjResults:
    """Numerical p_inj history and its empirical power-law fit:
    p_inj(t) ~ A_emp * t**alpha_emp."""

    t: np.ndarray[OneD, Float64]
    p_inj_num: np.ndarray[OneD, Float64]
    A_emp: float
    alpha_emp: float

    def p_inj_fit_emp(self):
        return self.A_emp * self.t**self.alpha_emp


@dataclass
class AnalyticalPInjResults(PInjResults):
    """PInjResults plus an analytical asymptotic prediction:
    p_inj(t) ~ A_ana * t**alpha_ana."""

    A_ana: float
    alpha_ana: float

    def p_inj_fit_ana(self):
        return self.A_ana * self.t**self.alpha_ana


# --- Aperture regimes -----------------------------------------------------

RIGID_APERTURE_EXPONENT: float = 0.5
RIGID_SIMILARITY_DOMAIN: float = 10.0
RIGID_SIMILARITY_POINTS: int = 200
SOFT_APERTURE_EXPONENT: float = 0.2


def _rigid_prefactor(run: file_io.RunData) -> float:
    """Rigid-aperture asymptotic prefactor. See [ref/derivation]."""
    zeta = np.linspace(0, RIGID_SIMILARITY_DOMAIN, RIGID_SIMILARITY_POINTS)

    theta = solve_linear_diffusion_const_flux(zeta)

    diffusivity = physics.diffusivity(run.params)
    if diffusivity <= 0:
        raise ValueError(
            f"physics.diffusivity(run.params) must be positive, got {diffusivity}"
        )

    return (
        theta[0]
        * run.params.q_0
        * run.params.k_n
        / diffusivity**RIGID_APERTURE_EXPONENT
    )


def _soft_prefactor(run: file_io.RunData) -> float:
    """Soft-aperture asymptotic prefactor. See [ref/derivation]."""
    w_final_inj = run.w[-1][0]
    if w_final_inj <= run.params.w_i:
        raise ValueError(
            "Final injection-point aperture "
            f"({w_final_inj}) is not larger than w_i "
            f"({run.params.w_i}); run may not have reached the soft regime."
        )

    a_param = physics.parameter_a(run.params)
    if a_param <= 0 or run.params.q_0 == 0:
        raise ValueError(
            "q_0 and physics.parameter_a(run.params) must be nonzero/positive "
            f"for fractional exponentiation, got q_0={run.params.q_0}, "
            f"parameter_a={a_param}"
        )

    theta = solve_neumann_n3(run.params.w_i / w_final_inj)[1]

    return (
        theta[0]
        * run.params.k_n
        * (run.params.q_0**2 / a_param) ** SOFT_APERTURE_EXPONENT
    )


def _empirical_fit(t: Vector, p_inj: np.ndarray) -> tuple[float, float]:
    return physics.fit_front_power_law(t, p_inj)


def _analyze_with(
    run: file_io.RunData,
    prefactor_fn: Callable[[file_io.RunData], float],
    exponent: float,
    slc: Optional[slice] = None,
) -> AnalyticalPInjResults:
    t, p_inj = find_p_inj(run)
    if slc is not None:
        t, p_inj = t[slc], p_inj[slc]

    A_emp, alpha_emp = _empirical_fit(t, p_inj)
    A_ana = prefactor_fn(run)

    return AnalyticalPInjResults(
        t=t,
        p_inj_num=p_inj,
        A_emp=A_emp,
        alpha_emp=alpha_emp,
        A_ana=A_ana,
        alpha_ana=exponent,
    )


def analyze_rigid(run: file_io.RunData) -> AnalyticalPInjResults:
    return _analyze_with(run, _rigid_prefactor, RIGID_APERTURE_EXPONENT)


def analyze_soft(run: file_io.RunData) -> AnalyticalPInjResults:
    return _analyze_with(run, _soft_prefactor, SOFT_APERTURE_EXPONENT)


def _default_slice(run: file_io.RunData, early: bool) -> slice:
    """Resolve the default early/late time slice for a run."""
    return physics.time_slice(run, early=early)


def analyze_early(
    run: file_io.RunData, slc: Optional[slice] = None
) -> AnalyticalPInjResults:
    """Analyze early-time (rigid-regime) p_inj behavior.

    If `slc` is None, an appropriate early-time slice is chosen
    automatically via physics.time_slice. Pass slice(None) explicitly to
    force use of the *entire* time range with no automatic slicing.
    """
    if slc is None:
        slc = _default_slice(run, early=True)
    return _analyze_with(run, _rigid_prefactor, RIGID_APERTURE_EXPONENT, slc=slc)


def analyze_late(
    run: file_io.RunData, slc: Optional[slice] = None
) -> AnalyticalPInjResults:
    """Analyze late-time (soft-regime) p_inj behavior.

    If `slc` is None, an appropriate late-time slice is chosen
    automatically via physics.time_slice. Pass slice(None) explicitly to
    force use of the *entire* time range with no automatic slicing.
    """
    if slc is None:
        slc = _default_slice(run, early=False)
    return _analyze_with(run, _soft_prefactor, SOFT_APERTURE_EXPONENT, slc=slc)


def analyze(run: file_io.RunData, slc: Optional[slice] = None) -> PInjResults:
    """Empirical power-law fit only — no analytical model evaluated."""
    t, p_inj = find_p_inj(run)
    if slc is not None:
        t, p_inj = t[slc], p_inj[slc]

    A_emp, alpha_emp = _empirical_fit(t, p_inj)
    return PInjResults(t=t, p_inj_num=p_inj, A_emp=A_emp, alpha_emp=alpha_emp)
