from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

Pressure = Callable[[np.ndarray], np.ndarray]


def nodes(n: int) -> tuple[np.ndarray, np.ndarray]:
    j = np.arange(1, n + 1)
    i = np.arange(1, n)
    s = np.cos(np.pi * (j - 1 / 2) / n)
    x = np.cos(np.pi * i / n)
    return s, x


def crack_length(p: Pressure, T: float, s: np.ndarray, bracket: tuple = (1e-8, 1e4)) -> float:
    return brentq(lambda a: np.mean(p(a * s)) - T, *bracket)


def slip_gradient(p: Pressure, a: float, s: np.ndarray, x: np.ndarray) -> np.ndarray:
    # bounded (K = 0) inversion: delta'(x) = sqrt(1 - x^2) phi(x), with
    # phi(x) = -(1/pi) PV int p(a s) / (sqrt(1 - s^2) (x - s)) ds, exact at the x_i
    n = len(s)
    return -1 / n * np.sum(p(a * s) / (x[:, None] - s), axis=1)


def slip(x: np.ndarray, phi: np.ndarray) -> np.ndarray:
    # delta(s_j) = int_{-1}^{s_j} sqrt(1 - x^2) phi dx by the midpoint rule in theta;
    # x_i <= s_j  <=>  i >= j, so this is a reversed cumulative sum (delta(s_n) ~ 0)
    n = len(x) + 1
    terms = np.pi / n * (1 - x**2) * phi
    return np.append(np.cumsum(terms[::-1])[::-1], 0.0)


@dataclass
class SlipSolution:
    p: Pressure
    T: float
    a: float  # crack half-length, in the length unit of p's argument
    s: np.ndarray  # first-kind nodes, where delta is given
    delta: np.ndarray  # delta(s_j) G' / (f dp a)
    x: np.ndarray  # second-kind nodes, where phi is given
    phi: np.ndarray  # delta'(x) = sqrt(1 - x^2) phi(x)

    def stress(self, xq) -> np.ndarray:
        """(tau - tau0) / (f dp) at xq (units of a)."""
        xq = np.atleast_1d(np.asarray(xq, float))
        out = self.T - self.p(self.a * xq)  # on the crack: the friction condition
        off = np.abs(xq) > 1
        # off the crack the integral is regular: Gauss rule of the second kind,
        # loses accuracy within a few node spacings of the tips
        n = len(self.x) + 1
        ddelta = (1 - self.x**2) * self.phi
        out[off] = -1 / n * np.sum(ddelta / (xq[off, None] - self.x), axis=1)
        return out


def solve_slip(
    p: Pressure, T: float, n: int = 128, bracket=(1e-8, 1e4)
) -> SlipSolution:
    s, x = nodes(n)
    a = crack_length(p, T, s, bracket)
    phi = slip_gradient(p, a, s, x)
    return SlipSolution(p, T, a, s, slip(x, phi), x, phi)
