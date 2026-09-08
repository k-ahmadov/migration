"""Small numerical helpers: power-law fitting and crossover functions."""

from functools import partial

import numpy as np
from scipy.optimize import curve_fit

from fracinj.types import Vector


def power_law(t, A, alpha):
    return A * t**alpha


def fit_power_law(
    t: Vector, y: Vector, *, p0_alpha: float = 0.8
) -> tuple[float, float]:
    """Least-squares fit of ``y = A * t**alpha``; returns ``(A, alpha)``."""
    (A, alpha), _ = curve_fit(
        power_law,
        t,
        y,
        p0=(np.max(y), p0_alpha),
        maxfev=20_000,
    )
    return A, alpha


def fit_power_law_fixed_exponent(t: Vector, y: Vector, exponent: float) -> float:
    """Fit only the prefactor ``A`` of ``y = A * t**exponent``."""
    (A,), _ = curve_fit(partial(power_law, alpha=exponent), t, y, p0=(np.max(y),))
    return A


def crossover(
    x: Vector,
    a: float,
    alpha: float,
    b: float,
    beta: float,
    x0: float,
    delta: float,
) -> Vector:
    """Smoothly join two power laws ``a*x**alpha`` and ``b*x**beta`` at ``x0``.

    ``delta`` controls how sharp the transition is.
    """
    return (a * x**alpha) / (1 + (x / x0) ** delta) + (b * x**beta) / (
        1 + (x0 / x) ** delta
    )


def fit_crossover_to_data(
    t: Vector,
    d: Vector,
    alpha_early: float,
    alpha_late: float,
    delta: float,
    t_c: float,
    p0: list[float],
) -> tuple[list[float], np.ndarray]:
    """
    Fit a two-regime power-law crossover model to data by optimizing
    the prefactors (a, b), holding the exponents, crossover scale,
    and sharpness fixed.

    Inputs:
        t: Independent variable (e.g. time).
        d: Dependent variable / data to fit against.
        alpha_early, alpha_late: Fixed power-law exponents for the
            early- and late-time regimes.
        delta: Sharpness of the crossover transition.
        t_c: Crossover (transition) scale.
        p0: Initial guess for [a, b].

    Returns:
        popt: Best-fit [a, b].
        perr: 1-sigma uncertainties on [a, b].
    """

    def crossover_fit(t, a_early, a_late):
        return crossover(
            t,
            a=a_early,
            alpha=alpha_early,
            b=a_late,
            beta=alpha_late,
            x0=t_c,
            delta=delta,
        )

    popt, pcov = curve_fit(crossover_fit, t, d, p0=p0, maxfev=10000)
    perr = np.sqrt(np.diag(pcov))

    return popt, perr
