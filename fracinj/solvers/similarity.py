"""Similarity (semi-analytical) solutions for pressure-dependent aperture flow.

The governing equation (Murphy et al., 2004, eq. 22) is

    a * d/dx [ w**n * dw/dx ] = dw/dt

with regime-dependent exponent ``n`` (0: rigid, 1: joint in contact,
3: joint liftoff). Reducing it with the similarity variable ``zeta``
gives an ODE that, after the substitution ``u = theta**(n+1)``
(Appendix A) to regularise the front singularity, is integrated with
RK45 and closed with a shooting method (``scipy.optimize.newton``) on
the far-field condition ``theta(zeta -> inf) = theta_inf``.

Reference:
    Murphy, H., Huang, C., Dash, Z., Zyvoloski, G., & White, A. (2004).
    Semianalytical solutions for fluid flow in rock joints with
    pressure-dependent openings. Water Resources Research, 40, W12506.
    https://doi.org/10.1029/2004WR003005
"""

from typing import Literal

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import newton
from scipy.special import erf

BC = Literal["dirichlet", "neumann"]

_ZETA_MAX = 5.0
_NEWTON_GUESS = {"dirichlet": -1.0, "neumann": 3.26}  # on u'(0) / u(0) respectively


def _integrate(rhs, y0, *, zeta_max: float, n_points: int) -> tuple[np.ndarray, np.ndarray]:
    zeta = np.linspace(0, zeta_max, n_points)
    sol = solve_ivp(
        rhs, (0, zeta_max), y0, t_eval=zeta, method="RK45", atol=1e-8, rtol=1e-6
    )
    return sol.t, sol.y


def _invert(u: np.ndarray, exponent: float) -> np.ndarray:
    """Undo ``u = theta**(1/exponent)``; clip negative numerical overshoot."""
    return np.maximum(u, 0.0) ** exponent


def solve_similarity(
    n: int,
    bc: BC,
    theta_inf: float = 0.0,
    *,
    zeta_max: float = _ZETA_MAX,
    n_points: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """Similarity profile ``theta(zeta)`` for exponent ``n`` and boundary condition ``bc``.

    Returns ``(zeta, theta)``.
    """
    if n == 0 and bc == "dirichlet":
        zeta = np.linspace(0, 10, 100)
        return zeta, (theta_inf - 1) * erf(zeta / 2) + 1

    exponent = 1 / (n + 1)  # theta = u ** exponent
    floor = 1e-8 if n >= 3 else 1e-5

    if bc == "dirichlet":
        # u'' = -zeta * u' / (2 * u**(n/(n+1))),  u(0) = 1, shoot on u'(0)
        def rhs(zeta, u):
            u0 = max(u[0], floor)
            return [u[1], -zeta * u[1] / (2 * u0 ** (n * exponent))]

        y0_of = lambda guess: [1.0, guess]
        target = theta_inf ** (n + 1)
    elif bc == "neumann":
        # u'' = c * (u**e - zeta * u**(e-1) * u'),  u'(0) = -(n+1), shoot on u(0)
        c = (n + 1) / (n + 2)

        def rhs(zeta, u):
            u0 = max(u[0], floor)
            return [u[1], c * (u0**exponent - zeta * u0 ** (exponent - 1) * u[1])]

        y0_of = lambda guess: [guess, -(n + 1)]
        target = theta_inf ** (n + 1)
    else:  # pragma: no cover - guarded by the Literal type
        raise ValueError(f"unknown boundary condition {bc!r}")

    def objective(guess):
        _, y = _integrate(rhs, y0_of(guess), zeta_max=zeta_max, n_points=n_points)
        return y[0, -1] - target

    guess = newton(objective, _NEWTON_GUESS[bc])
    zeta, y = _integrate(rhs, y0_of(guess), zeta_max=zeta_max, n_points=n_points)
    return zeta, _invert(y[0], exponent)


# --- Named convenience wrappers -------------------------------------------------


def solve_dirichlet_n0(theta_inf: float = 0.0):
    return solve_similarity(0, "dirichlet", theta_inf)


def solve_dirichlet_n1(theta_inf: float = 0.0):
    return solve_similarity(1, "dirichlet", theta_inf)


def solve_dirichlet_n3(theta_inf: float = 0.0):
    return solve_similarity(3, "dirichlet", theta_inf, n_points=500)


def solve_neumann_n1(theta_inf: float = 0.0):
    return solve_similarity(1, "neumann", theta_inf)


def solve_neumann_n3(theta_inf: float = 0.0):
    return solve_similarity(3, "neumann", theta_inf)
