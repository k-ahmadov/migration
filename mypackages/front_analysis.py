from dataclasses import dataclass

import numpy as np

from mypackages import file_io, front_detection, physics
from mypackages.typesdefs import (
    Float64,
    OneD,
    Vector,
)

# %% --- Case analysis ----------------------------------------------------------


@dataclass
class FrontResults:
    t_front: Vector
    x_front: Vector
    A_emp: float
    alpha_emp: float

    def x_empirical(self) -> Vector:
        return self.A_emp * self.t_front**self.alpha_emp

    def calculate_local_exponent(
        self,
        window: int = 20,
    ) -> tuple[Vector, np.ndarray[OneD, Float64]]:
        valid = (self.t_front > 0) & (self.x_front > 0)
        x, y = self.t_front[valid], self.x_front[valid]
        log_x = np.log(x)
        log_y = np.log(y)
        n = len(x)
        slope = np.full(n, np.nan)
        half = window // 2
        for i in range(half, n - half):
            s = slice(i - half, i + half)
            slope[i], _ = np.polyfit(log_x[s], log_y[s], deg=1)
        return self.t_front[valid], slope

    def calculate_velocity(self) -> np.ndarray[OneD, Float64]:
        distance = self.x_front[-1] - self.x_front[0]
        duration = self.t_front[-1] - self.t_front[0]
        return distance / duration


@dataclass
class AnalyticalFrontResults(FrontResults):
    A_ana: float
    alpha_ana: float
    zeta_front: float

    def x_analytical(self) -> np.ndarray:
        return self.A_ana * self.t_front**self.alpha_ana


def analyze(
    run: file_io.RunData,
    threshold: Vector | None = None,
    slc: slice = slice(None),
    stress_front: bool = False,
) -> FrontResults:
    if stress_front:
        x_front, idx_cut = front_detection.find_stress_front(
            run.x_sc, run.sn[slc], L=run.params.L
        )
        t_front = run.t[slc][:idx_cut]
    elif threshold is not None:
        x_front, has_crossing = front_detection.find_field_front(
            run.x_sc, run.p[slc], threshold
        )
        t_front = run.t[slc][has_crossing]
    else:
        raise ValueError("Either provide a threshold or set stress_front=True")

    A_emp, alpha_emp = physics.fit_front_power_law(t_front, x_front)
    return FrontResults(
        t_front=t_front, x_front=x_front, A_emp=A_emp, alpha_emp=alpha_emp
    )


def _compute_zeta_front_rigid(
    x_front: np.ndarray, t_front: np.ndarray, D: float, alpha: float
) -> float:
    """Estimate the dimensionless front amplitude zeta from front positions."""
    mask = t_front != 0
    if not np.any(mask):
        raise ValueError("No non-zero front times — cannot estimate zeta.")
    return float(np.mean(x_front[mask] / (D * t_front[mask]) ** alpha))


RIGID_DIFFUSION_EXPONENT: float = 0.5


def analyze_rigid(
    run: file_io.RunData,
    threshold: Vector | None = None,
    slc: slice = slice(None),
    stress_front: bool = False,
) -> AnalyticalFrontResults:
    result = analyze(run, threshold, slc, stress_front)
    D = physics.diffusivity(run.params)
    alpha_ana = RIGID_DIFFUSION_EXPONENT
    zeta = _compute_zeta_front_rigid(result.x_front, result.t_front, D, alpha_ana)
    return AnalyticalFrontResults(
        **vars(result), A_ana=zeta * D**alpha_ana, alpha_ana=alpha_ana, zeta_front=zeta
    )


def _compute_zeta_front_soft(
    x_front: np.ndarray, t_front: np.ndarray, a: float, q: float, alpha: float
) -> float:
    """Estimate the dimensionless front amplitude zeta from front positions."""
    mask = t_front != 0
    if not np.any(mask):
        raise ValueError("No non-zero front times — cannot estimate zeta.")
    denominator = a ** (1 / 5) * q ** (3 / 5) * t_front[mask] ** (alpha)
    zeta_front = float(np.mean(x_front[mask] / denominator))
    return zeta_front


SOFT_DIFFUSION_EXPONENT: float = 0.8


def analyze_soft(
    run: file_io.RunData,
    threshold: Vector | None = None,
    slc: slice = slice(None),
    stress_front: bool = False,
) -> AnalyticalFrontResults:
    result = analyze(run, threshold, slc, stress_front)
    q = run.params.q_0
    a = physics.parameter_a(run.params)
    alpha_ana = SOFT_DIFFUSION_EXPONENT
    zeta = _compute_zeta_front_soft(result.x_front, result.t_front, a, q, alpha_ana)
    A_ana = zeta * (a * q**3) ** (1 / 5)
    return AnalyticalFrontResults(
        **vars(result), A_ana=A_ana, alpha_ana=alpha_ana, zeta_front=zeta
    )


def analyze_early_time(
    run: file_io.RunData,
    pc: float | None = None,
    stress_front: bool = False,
    slc: slice = slice(None),
) -> AnalyticalFrontResults:
    if slc == slice(None):
        slc = physics.time_slice(run, early=True)
    if not stress_front:
        if pc is None:
            raise ValueError("pc must be provided when stress_front=False")
        threshold = front_detection.constant_pressure_threshold(run.t[slc], pc)
    else:
        threshold = None
    return analyze_rigid(run, threshold, slc, stress_front)


def analyze_late_time(
    run: file_io.RunData,
    pc: float | None = None,
    stress_front: bool = False,
    slc: slice = slice(None),
) -> AnalyticalFrontResults:
    if slc == slice(None):
        slc = physics.time_slice(run, early=True)
    if not stress_front:
        if pc is None:
            raise ValueError("pc must be provided when stress_front=False")
        threshold = front_detection.constant_pressure_threshold(run.t[slc], pc)
    else:
        threshold = None
    return analyze_soft(run, threshold, slc, stress_front)
