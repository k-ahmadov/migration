"""
Similarity solutions for the nonlinear diffusion equation governing
fluid flow in rock joints with pressure-dependent aperture.

The governing equation (Murphy et al., 2004, eq. 22) is:

    a · ∂/∂x [ wⁿ · ∂w/∂x ] = ∂w/∂t

where w is the joint aperture, and the coefficient a and exponent n
depend on the joint regime:

    n=0: linear diffusion,    a = k_n wᵢ³ / (12μ)   (rigid)
    n=1: joint in contact,    a = k_n wᵢ² / (12μ)   (semi-rigid)
    n=3: joint liftoff,       a = k_n  / (12μ)      (soft)

The PDE is reduced to an ODE via the similarity variable zeta = x/δ, where
the propagation scale δ depends on the boundary condition:

    Dirichlet (constant aperture w₀ at x=0):
        δ = (a·w₀ⁿ·t)^(1/2)     (eq. 23, Murphy et al., 2004)

    Neumann (constant flux q at x=0):
        δ = (a·qⁿ·t^(n+1))^(1/(n+2))   (eq. 29, Murphy et al., 2004)

The substitution u = theta^(n+1) (Appendix A, Murphy et al., 2004) is applied
to regularize the singularity at theta=0, transforming the ODE into a
smoother first-order system integrated with RK45. The solution is
recovered via theta = u^(1/(n+1)).

The shooting method (scipy.newton) finds the unknown initial condition
(either u'(0) for Dirichlet or u(0) for Neumann) such that the far-field
boundary condition theta(zeta→∞) = theta_inf is satisfied.

Reference:
    Murphy, H., Huang, C., Dash, Z., Zyvoloski, G., & White, A. (2004).
    Semianalytical solutions for fluid flow in rock joints with
    pressure-dependent openings. Water Resources Research, 40, W12506.
    https://doi.org/10.1029/2004WR003005
"""

from typing import Callable

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import newton
from scipy.special import erf

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _integrate(
    F: Callable, zeta_span: tuple[float, float], y0: list[float], n_points: int = 100
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate the ODE system F over zeta_span and return (zeta, u)."""
    zeta_eval = np.linspace(*zeta_span, n_points)
    sol = solve_ivp(F, zeta_span, y0, t_eval=zeta_eval, method="RK45", atol=1e-8, rtol=1e-6)
    return sol.t, sol.y


def _invert(u: np.ndarray, exponent: float) -> np.ndarray:
    """
    Invert the substitution u = theta^(1/exponent) → theta = max(u, 0)^exponent.

    The substitution u = theta^(n+1) is introduced in Appendix A of Murphy et al.
    (2004) to remove the singularity of the ODE at theta=0. max(u, 0) clips
    any small negative numerical overshoot near the propagation front.
    """
    return np.maximum(u, 0.0) ** exponent


# ---------------------------------------------------------------------------
# Dirichlet BC  (constant aperture at x=0)
# ---------------------------------------------------------------------------


def solve_dirichlet_n0(theta_inf: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """
    Exact similarity solution for n=0 (linear diffusion), Dirichlet BC.

    Solves the linear diffusion limit of eq. (22) in Murphy et al. (2004):

        ∂²w/∂x² = (1/κ) · ∂w/∂t,   κ = w²/(12μβ)

    with w(0,t) = w₀, w(x,0) = wᵢ, w(∞,t) = wᵢ.

    The exact solution in similarity form (eq. 25, Murphy et al., 2004) is:

        theta(zeta) = erfc(zeta/2),   zeta = x / √(κt)

    For arbitrary initial aperture theta_inf = wᵢ/w₀:

        theta(zeta) = (theta_inf - 1) · erf(zeta/2) + 1

    Args:
        theta_inf: Far-field dimensionless aperture wᵢ/w₀. Default 0 (zero
               initial aperture, Table 1 of Murphy et al., 2004).

    Returns:
        zeta: Similarity variable array.
        theta: Dimensionless aperture w/w₀.
    """
    zeta = np.linspace(0, 10, 100)
    theta = (theta_inf - 1) * erf(zeta / 2) + 1
    return zeta, theta


def solve_dirichlet_n1(theta_inf: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """
    Numerical similarity solution for n=1 (joint in contact), Dirichlet BC.

    Solves eq. (23) of Murphy et al. (2004) for n=1:

        d/dzeta [ theta · dtheta/dzeta ] + (zeta/2) · dtheta/dzeta = 0

    with theta(0) = 1, theta(∞) = theta_inf.

    Substitution u = theta² (Appendix A) gives the integrated system:

        u'' = -zeta · u' / (2 · u^(1/2))

    Shooting over u'(0) to match u(∞) = theta_inf².

    Args:
        theta_inf: Far-field dimensionless aperture wᵢ/w₀.

    Returns:
        zeta: Similarity variable array.
        theta: Dimensionless aperture w/w₀.
    """
    zeta_span = (0, 5)

    def F(zeta, u):
        u0, u1 = u
        u0 = max(u0, 1e-5)
        return [u1, -zeta * u1 / (2 * u0**0.5)]

    def objective(u1_0):
        _, y = _integrate(F, zeta_span, [1.0, u1_0])
        return y[0, -1] - theta_inf**2

    u1_0 = newton(objective, -1.0)
    zeta, y = _integrate(F, zeta_span, [1.0, u1_0])
    return zeta, _invert(y[0], 0.5)


def solve_dirichlet_n3(theta_inf: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """
    Numerical similarity solution for n=3 (joint liftoff), Dirichlet BC.

    Solves eq. (23) of Murphy et al. (2004) for n=3:

        d/dzeta [ theta³ · dtheta/dzeta ] + (zeta/2) · dtheta/dzeta = 0

    with theta(0) = 1, theta(∞) = theta_inf.

    Substitution u = theta⁴ (Appendix A) gives:

        u'' = -zeta · u' / (2 · u^(3/4))

    The solution exhibits sharp shock-like front at zeta ≈ 0.875 for theta_inf=0
    (Table 1, Murphy et al., 2004), characteristic of nonlinear diffusion.

    Args:
        theta_inf: Far-field dimensionless aperture wᵢ/w₀.

    Returns:
        zeta: Similarity variable array.
        theta: Dimensionless aperture w/w₀.
    """
    zeta_span = (0, 5)

    def F(zeta, u):
        u0, u1 = u
        u0 = max(u0, 1e-5)
        return [u1, -zeta * u1 / (2 * u0**0.75)]

    def objective(u1_0):
        _, y = _integrate(F, zeta_span, [1.0, u1_0], n_points=500)
        return y[0, -1] - theta_inf**4

    u1_0 = newton(objective, -1.0)
    zeta, y = _integrate(F, zeta_span, [1.0, u1_0], n_points=500)
    return zeta, _invert(y[0], 0.25)


# ---------------------------------------------------------------------------
# Neumann BC  (constant flux at x=0)
# ---------------------------------------------------------------------------


def solve_neumann_n1(theta_inf: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """
    Numerical similarity solution for n=1 (joint in contact), Neumann BC.

    Solves eq. (30) of Murphy et al. (2004) for n=1:

        d/dzeta [ theta · dtheta/dzeta ] + (2/3) · zeta · dtheta/dzeta - theta/3 = 0

    with boundary conditions (eq. 31):

        theta(0) · dtheta/dzeta|₀ = -1,   theta(∞) = theta_inf

    Similarity variable: zeta = x / (a·q·t²)^(1/3)   [eq. 28-29, n=1]

    Substitution u = theta² gives:

        u'' = (2/3) · (u^(1/2) - zeta · u^(-1/2) · u')

    Shooting over u(0) with fixed u'(0) = -2 (from Neumann BC).

    Args:
        theta_inf: Far-field dimensionless aperture.

    Returns:
        zeta: Similarity variable array.
        theta: Dimensionless aperture.
    """
    zeta_span = (0, 5)
    u1_0 = -2.0  # fixed by Neumann BC: thetaⁿ · dtheta/dzeta|₀ = -1 → u'(0) = -2

    def F(zeta, u):
        u0, u1 = u
        u0 = max(u0, 1e-5)
        return [u1, (2 / 3) * (u0**0.5 - zeta * u0**-0.5 * u1)]

    def objective(u0):
        _, y = _integrate(F, zeta_span, [u0, u1_0])
        return y[0, -1] - theta_inf**2

    u0 = newton(objective, 3.26)
    zeta, y = _integrate(F, zeta_span, [u0, u1_0])
    return zeta, _invert(y[0], 0.5)


def solve_neumann_n3(theta_inf: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """
    Numerical similarity solution for n=3 (joint liftoff), Neumann BC.

    Solves eq. (30) of Murphy et al. (2004) for n=3:

        d/dzeta [ theta³ · dtheta/dzeta ] + (4/5) · zeta · dtheta/dzeta - 2theta/5 = 0

    with boundary conditions (eq. 31):

        theta³(0) · dtheta/dzeta|₀ = -1,   theta(∞) = theta_inf

    Similarity variable: zeta = x / (a·q³·t⁴)^(1/5)   [eq. 28-29, n=3]

    Substitution u = theta⁴ gives:

        u'' = 0.8 · (u^(1/4) - zeta · u^(-3/4) · u')

    Shooting over u(0) with fixed u'(0) = -4 (from Neumann BC).
    The prefactor 0.8 = (n+1)/(n+2) evaluated at n=3.

    Args:
        theta_inf: Far-field dimensionless aperture.

    Returns:
        zeta: Similarity variable array.
        theta: Dimensionless aperture.
    """
    zeta_span = (0, 5)
    u1_0 = -4.0  # fixed by Neumann BC: thetaⁿ · dtheta/dzeta|₀ = -1 → u'(0) = -(n+1) = -4

    def F(zeta, u):
        u0, u1 = u
        u0 = max(u0, 1e-8)  # tighter floor: u^(-3/4) more singular than n=1
        return [u1, 0.8 * (u0**0.25 - zeta * u0**-0.75 * u1)]

    def objective(u0):
        _, y = _integrate(F, zeta_span, [u0, u1_0])
        return y[0, -1] - theta_inf**4

    u0 = newton(objective, 3.26)
    zeta, y = _integrate(F, zeta_span, [u0, u1_0])
    return zeta, _invert(y[0], 0.25)
