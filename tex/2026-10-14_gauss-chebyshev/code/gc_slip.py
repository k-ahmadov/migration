"""Gauss-Chebyshev solver for fault slip driven by a pore-pressure perturbation.

Solves, for the slip delta(x) and the crack half-length a,

    T - p(a x) = (1/pi) PV int_{-1}^{1} delta'(s) / (s - x) ds,    |x| < 1,

with delta(+-1) = 0 and bounded stress at both tips (K = 0). The slip gradient is
written delta'(s) = sqrt(1 - s^2) F(s); F is sought at the roots s_j of U_n and
the equation is collocated at the roots x_i of T_{n+1} (Viesca & Garagash 2018,
JMPS 113, Sec. 2 and 3.2). The derivation is in ../main.tex.

The '# begin:<name>' / '# end:<name>' comments delimit the excerpts that main.tex
pulls in with \\lstinputlisting; keep them when editing.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

# p(xi) -> (p - p0) / Delta p, with xi the along-fault position in any length unit
Pressure = Callable[[np.ndarray], np.ndarray]


# begin:nodes
def nodes(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Quadrature nodes s_j and weights w_j (n each), collocation points x_i (n + 1)."""
    j = np.arange(1, n + 1)
    s = np.cos(np.pi * j / (n + 1))  # roots of U_n
    w = np.pi / (n + 1) * (1 - s**2)  # Gauss weights for the weight sqrt(1 - s^2)
    i = np.arange(1, n + 2)
    x = np.cos(np.pi * (i - 0.5) / (n + 1))  # roots of T_{n+1}
    return s, w, x
# end:nodes


# begin:hilbert
def hilbert_matrix(s: np.ndarray, w: np.ndarray, x: np.ndarray) -> np.ndarray:
    """H_ij with  (1/pi) PV int sqrt(1 - s^2) F(s) / (x_i - s) ds = sum_j H_ij F(s_j)."""
    return w[None, :] / (np.pi * (x[:, None] - s[None, :]))
# end:hilbert


# begin:crack_length
def crack_length(p: Pressure, T: float, x: np.ndarray, bracket=(1e-8, 1e4)) -> float:
    """Half-length a from the discrete K = 0 condition  T = mean_i p(a x_i)."""
    return brentq(lambda a: np.mean(p(a * x)) - T, *bracket)
# end:crack_length


# begin:coefficients
def coefficient_matrix(s: np.ndarray) -> np.ndarray:
    """B_kj with  A_k = sum_j B_kj F(s_j)  for  F(s) = sum_{k<n} A_k U_k(s)."""
    n = len(s)
    th = np.arccos(s)
    k = np.arange(n)[:, None]
    return 2 / (n + 1) * np.sin(th) * np.sin((k + 1) * th)
# end:coefficients


# begin:integration
def integration_matrix(s: np.ndarray, xq) -> np.ndarray:
    """S_qj with  delta(xq_q) = sum_j S_qj F(s_j),  taking delta(-1) = 0."""
    n = len(s)
    th = np.arccos(np.clip(np.atleast_1d(np.asarray(xq, float)), -1, 1))[:, None]
    k = np.arange(1, n)
    # int_{-1}^{x} sqrt(1 - s^2) U_k(s) ds, columns k = 0, 1, ..., n-1
    Phi = np.hstack([
        0.5 * (np.pi - th + 0.5 * np.sin(2 * th)),
        0.5 * (np.sin((k + 2) * th) / (k + 2) - np.sin(k * th) / k),
    ])
    return Phi @ coefficient_matrix(s)
# end:integration


# begin:stress
def stress_change(A: np.ndarray, xq) -> np.ndarray:
    """Delta tau(x) = (1/pi) PV int delta'(s) / (s - x) ds, on and off the crack."""
    xq = np.atleast_1d(np.asarray(xq, float))
    m = np.arange(1, len(A) + 1)[:, None]  # T_{k+1}(x) inside, rho(x)^{k+1} outside
    on = np.abs(xq) <= 1
    out = np.empty_like(xq)
    out[on] = -A @ np.cos(m * np.arccos(xq[on]))
    xo = xq[~on]
    rho = xo - np.sign(xo) * np.sqrt(xo**2 - 1)
    out[~on] = -A @ rho**m
    return out
# end:stress


# begin:solve
@dataclass
class SlipSolution:
    T: float
    a: float  # crack half-length, in the length unit of p's argument
    s: np.ndarray  # quadrature nodes (roots of U_n)
    F: np.ndarray  # F(s_j); slip gradient delta'(s) = sqrt(1 - s^2) F(s)
    residual: float  # max_i |(H F)_i - (p(a x_i) - T)|

    def slip(self, x) -> np.ndarray:
        """delta at x in [-1, 1] (x in units of a, delta in units of f Dp a / G')."""
        return integration_matrix(self.s, x) @ self.F

    def stress(self, x) -> np.ndarray:
        """Delta tau at any x (units of a), in units of f Dp."""
        return stress_change(coefficient_matrix(self.s) @ self.F, x)


def solve_slip(p: Pressure, T: float, n: int = 128, bracket=(1e-8, 1e4)) -> SlipSolution:
    s, w, x = nodes(n)
    a = crack_length(p, T, x, bracket)  # 1) K = 0 fixes the half-length
    H = hilbert_matrix(s, w, x)
    rhs = p(a * x) - T
    F = np.linalg.lstsq(H, rhs, rcond=None)[0]  # 2) n + 1 consistent equations, n unknowns
    return SlipSolution(T, a, s, F, np.max(np.abs(H @ F - rhs)))
# end:solve
