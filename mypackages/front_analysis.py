from dataclasses import dataclass

import numpy as np

from mypackages import file_io, front_detection, physics
from mypackages.types import (
    CriticalPressure,
    Float64,
    FrontDetectionThreshold,
    FrontPositions,
    OneD,
    Prefactor,
    ScalingExponent,
    Time,
    XPositions,
)

# %% --- Case analysis ----------------------------------------------------------


@dataclass
class FrontResults:
    t_front: Time
    x_front: FrontPositions
    A_emp: Prefactor
    alpha_emp: ScalingExponent

    def x_empirical(self) -> XPositions:
        return self.A_emp * self.t_front**self.alpha_emp

    def calculate_local_exponent(
        self,
        window: int = 20,
    ) -> tuple[Time, np.ndarray[OneD, Float64]]:
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
    A_ana: Prefactor
    alpha_ana: ScalingExponent
    zeta_front: float

    def x_analytical(self) -> np.ndarray:
        return self.A_ana * self.t_front**self.alpha_ana


def analyze(
    run: file_io.RunData,
    threshold: FrontDetectionThreshold | None = None,
    slc: slice = slice(None),
    stress_front: bool = False,
) -> FrontResults:
    if stress_front:
        x_front, idx_cut = front_detection.find_stress_front(
            run.x_sc, run.sn[slc], mesh_size=2
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
    return FrontResults(t_front=t_front, x_front=x_front, A_emp=A_emp, alpha_emp=alpha_emp)


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
    threshold: FrontDetectionThreshold | None = None,
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
    x_front: np.ndarray, t_front: np.ndarray, M: float, q: float, alpha: float
) -> float:
    """Estimate the dimensionless front amplitude zeta from front positions."""
    mask = t_front != 0
    if not np.any(mask):
        raise ValueError("No non-zero front times — cannot estimate zeta.")
    denominator = M ** (1 / 5) * q ** (3 / 5) * t_front[mask] ** (alpha)
    zeta_front = float(np.mean(x_front[mask] / denominator))
    return zeta_front


SOFT_DIFFUSION_EXPONENT: float = 0.8


def analyze_soft(
    run: file_io.RunData,
    threshold: FrontDetectionThreshold | None = None,
    slc: slice = slice(None),
    stress_front: bool = False,
) -> AnalyticalFrontResults:
    result = analyze(run, threshold, slc, stress_front)
    q = run.params.q_0 or run.params.q
    assert q is not None and q > 0, f"expected correct injection rate, got {q}"
    M = physics.parameter_a(run.params)
    alpha_ana = SOFT_DIFFUSION_EXPONENT
    zeta = _compute_zeta_front_soft(result.x_front, result.t_front, M, q, alpha_ana)
    A_ana = zeta * (M * q**3) ** (1 / 5)
    return AnalyticalFrontResults(
        **vars(result), A_ana=A_ana, alpha_ana=alpha_ana, zeta_front=zeta
    )


def time_slice(run: file_io.RunData, early: bool) -> slice:
    t_c = physics.critical_time(run.params)
    idx = np.searchsorted(run.t, t_c) + 1
    # return slice(30 * idx) if early else slice(50 * idx, None)
    return slice(idx) if early else slice(2 * idx, None)


def analyze_early_time(
    run: file_io.RunData,
    pc: CriticalPressure | None = None,
    stress_front: bool = False,
    slc: slice = slice(None),
) -> AnalyticalFrontResults:
    if slc == slice(None):
        slc = time_slice(run, early=True)
    if not stress_front:
        if pc is None:
            raise ValueError("pc must be provided when stress_front=False")
        threshold = front_detection.constant_pressure_threshold(run.t[slc], pc)
    else:
        threshold = None
    return analyze_rigid(run, threshold, slc, stress_front)


def analyze_late_time(
    run: file_io.RunData,
    pc: CriticalPressure | None = None,
    stress_front: bool = False,
    slc: slice = slice(None),
) -> AnalyticalFrontResults:
    if slc == slice(None):
        slc = time_slice(run, early=True)
    if not stress_front:
        if pc is None:
            raise ValueError("pc must be provided when stress_front=False")
        threshold = front_detection.constant_pressure_threshold(run.t[slc], pc)
    else:
        threshold = None
    return analyze_soft(run, threshold, slc, stress_front)
