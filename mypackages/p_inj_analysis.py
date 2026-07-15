from dataclasses import dataclass
from typing import Callable
import numpy as np
from mypackages import file_io, physics
from mypackages.types import Float64, OneD, Time
from mysolvers.exact_solutions import solve_linear_diffusion_const_flux
from mysolvers.similarity_solutions import solve_neumann_n3


def find_p_inj(run: file_io.RunData) -> np.ndarray[OneD, Float64]:
    """Injection-point pressure history, sorted by x_sc."""
    _, p = file_io.sort_fields(run.x_sc, run.p)
    return p[:, 0]


@dataclass
class PInjResults:
    """Numerical p_inj history and its empirical power-law fit:
    p_inj(t) ~ A_emp * t**alpha_emp."""
    p_inj_num: np.ndarray[OneD, Float64]
    A_emp: float
    alpha_emp: float

    def p_inj_fit_emp(self, time: Time):
        return self.A_emp * time**self.alpha_emp


@dataclass
class AnalyticalPInjResults(PInjResults):
    """PInjResults plus an analytical asymptotic prediction:
    p_inj(t) ~ A_ana * t**alpha_ana."""
    A_ana: float
    alpha_ana: float

    def p_inj_fit_ana(self, time: Time):
        return self.A_ana * time**self.alpha_ana


# --- Aperture regimes -----------------------------------------------------
RIGID_APERTURE_EXPONENT: float = 0.5
RIGID_SIMILARITY_DOMAIN: float = 10.0
RIGID_SIMILARITY_POINTS: int = 200


def _rigid_prefactor(run: file_io.RunData) -> float:
    """Rigid-aperture asymptotic prefactor. See [ref/derivation]."""
    zeta = np.linspace(0, RIGID_SIMILARITY_DOMAIN, RIGID_SIMILARITY_POINTS)
    theta = solve_linear_diffusion_const_flux(zeta)
    return (
        theta[0]
        * run.params.flux
        * run.params.k_n
        / physics.diffusivity(run.params) ** RIGID_APERTURE_EXPONENT
    )


SOFT_APERTURE_EXPONENT: float = 0.2


def _soft_prefactor(run: file_io.RunData) -> float:
    """Soft-aperture asymptotic prefactor. See [ref/derivation]."""
    theta = solve_neumann_n3(run.params.w_i / run.w[-1][0])[1]
    return (
        theta[0]
        * run.params.k_n
        * (run.params.flux**2 / physics.parameter_a(run.params))
        ** SOFT_APERTURE_EXPONENT
    )


def _empirical_fit(run: file_io.RunData, p_inj: np.ndarray) -> tuple[float, float]:
    return physics.fit_front_power_law(run.t, p_inj)


def _analyze_with(
    run: file_io.RunData,
    prefactor_fn: Callable[[file_io.RunData], float],
    exponent: float,
) -> AnalyticalPInjResults:
    p_inj = find_p_inj(run)
    A_emp, alpha_emp = _empirical_fit(run, p_inj)
    A_ana = prefactor_fn(run)
    return AnalyticalPInjResults(
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


def analyze(run: file_io.RunData) -> PInjResults:
    """Empirical power-law fit only — no analytical model evaluated."""
    p_inj = find_p_inj(run)
    A_emp, alpha_emp = _empirical_fit(run, p_inj)
    return PInjResults(p_inj_num=p_inj, A_emp=A_emp, alpha_emp=alpha_emp)
