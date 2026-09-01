"""Small numerical helpers: power-law fitting and crossover functions."""

from functools import partial

import numpy as np
from scipy.optimize import curve_fit

from fracinj.types import Vector


def power_law(t, A, alpha):
    return A * t**alpha


def fit_power_law(t: Vector, y: Vector, *, p0_alpha: float = 0.8) -> tuple[float, float]:
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
