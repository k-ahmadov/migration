"""Similarity solutions of nonlinear diffusion with a power-law diffusivity.

Governing equation (dimensionless; for a * d/dx[...] = dw/dt use tau = a*t)

    (1/x^m) d/dx [ x^m  w^n  dw/dx ] = dw/dt,      x > 0

    n         diffusivity exponent: 0 rigid (linear), 1 joint in contact,
              3 joint liftoff / thin film (Murphy et al., 2004, eq. 22)
    m         geometry: 0 planar (1D), 1 radial (axisymmetric)

Boundary condition at the origin, growing in time like t^i:

    "dirichlet":   w(0, t) = t^i                     (planar only)
    "neumann":     -x^m w^n dw/dx -> t^i  as x -> 0  (injection flux)

    i = 0 gives a constant head / constant rate; i = 1 a linear ramp.

Far field: w -> theta_inf (theta_inf = 0: dry / closed joint ahead of a front).

Similarity form
---------------
    w = t^alpha f(zeta),   zeta = x / t^beta

    Dirichlet:  alpha = i,                              beta = (n*alpha + 1)/2
    Neumann:    alpha = (2i + 1 - m) / (n(m+1) + 2),    beta = (n*alpha + 1)/2

which reduces the PDE to the ODE (written as a first-order system)

    f' = g / (zeta^m f^n),      g' = zeta^m (alpha f - beta zeta f'),

where g = zeta^m f^n f' is the flux.  Dirichlet: f(0) = 1.  Neumann: g(0) = -1.

Solution methods (chosen automatically)
---------------------------------------
"front"    theta_inf = 0 and n > 0.  The solution has a finite front zeta_N.
           Start just behind a front placed at zeta = 1, using the local form
               f ~ [n beta zeta_N (zeta_N - zeta)]^(1/n),
           integrate inward to the origin once, then use the invariance
               f -> lam F,  zeta -> lam^(n/2) zeta
           to meet the origin condition exactly: no iteration.
"shooting" theta_inf > 0 or n = 0 (no front).  Integrate outward from the
           origin and bracket + bisect (brentq) on the unknown origin value
           so that f(zeta_max) = theta_inf.

Reference:
    Murphy, H., Huang, C., Dash, Z., Zyvoloski, G., & White, A. (2004).
    Semianalytical solutions for fluid flow in rock joints with
    pressure-dependent openings. Water Resources Research, 40, W12506.
    https://doi.org/10.1029/2004WR003005
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.integrate import solve_ivp, trapezoid
from scipy.optimize import brentq

BC = Literal["dirichlet", "neumann"]
Geometry = Literal["planar", "radial"]

# Numerical settings, all in one place
RTOL, ATOL = 1e-11, 1e-13
FRONT_OFFSET = 1e-6  # start this far behind the (unit) front
ZETA_MIN_RADIAL = 1e-8  # radial problems are singular at zeta = 0
ZETA_MAX_SHOOTING = 10.0  # default far-field location for shooting


# --------------------------------------------------------------------------
# Problem definition
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Problem:
    n: float  # diffusivity exponent, D ~ w^n
    bc: BC  # "dirichlet" or "neumann"
    i: float = 0.0  # boundary value / flux grows like t^i
    geometry: Geometry = "planar"  # "planar" (1D) or "radial"
    theta_inf: float = 0.0  # far-field value

    def __post_init__(self):
        if self.n < 0:
            raise ValueError("n must be >= 0")
        if self.bc not in ("dirichlet", "neumann"):
            raise ValueError(f"unknown boundary condition {self.bc!r}")
        if self.geometry not in ("planar", "radial"):
            raise ValueError(f"unknown geometry {self.geometry!r}")
        if self.geometry == "radial" and self.bc == "dirichlet":
            raise ValueError("a Dirichlet condition at a point (radial) is ill-posed")
        if self.theta_inf < 0:
            raise ValueError("theta_inf must be >= 0")

    @property
    def m(self) -> int:
        return 0 if self.geometry == "planar" else 1

    @property
    def alpha(self) -> float:
        if self.bc == "dirichlet":
            return self.i
        return (2 * self.i + 1 - self.m) / (self.n * (self.m + 1) + 2)

    @property
    def beta(self) -> float:
        return (self.n * self.alpha + 1) / 2

    @property
    def has_front(self) -> bool:
        return self.theta_inf == 0 and self.n > 0

    def rhs(self, zeta: float, y: np.ndarray) -> list[float]:
        """Reduced ODE as a first-order system in y = [f, g], g = zeta^m f^n f'."""
        f, g = y
        zm = zeta**self.m
        fp = g / (zm * f**self.n)
        return [fp, zm * (self.alpha * f - self.beta * zeta * fp)]


# --------------------------------------------------------------------------
# Result
# --------------------------------------------------------------------------
@dataclass
class Solution:
    problem: Problem
    zeta: np.ndarray  # grid the profile was sampled on
    theta: np.ndarray  # f(zeta) on that grid
    zeta_front: float | None  # front position (None if no front)
    method: str  # "front" or "shooting"
    _profile: Callable[[np.ndarray], np.ndarray]

    @property
    def alpha(self) -> float:
        return self.problem.alpha

    @property
    def beta(self) -> float:
        return self.problem.beta

    def __call__(self, zeta) -> np.ndarray:
        """Similarity profile f(zeta) at arbitrary points."""
        return self._profile(np.atleast_1d(np.asarray(zeta, dtype=float)))

    def __iter__(self):
        """Allows  zeta, theta = solve(...)  like the old functions."""
        return iter((self.zeta, self.theta))

    def w(self, x, t):
        """Physical solution w(x, t) = t^alpha f(x / t^beta)."""
        return t**self.alpha * self(np.asarray(x, dtype=float) / t**self.beta)

    def front(self, t):
        """Front position x_N(t) = zeta_N t^beta (None if no front)."""
        return None if self.zeta_front is None else self.zeta_front * t**self.beta

    def mass_check(self) -> tuple[float, float] | None:
        """Neumann, theta_inf = 0: int zeta^m f dzeta must equal 1/(i+1)."""
        p = self.problem
        if p.bc != "neumann" or p.theta_inf != 0:
            return None
        hi = self.zeta_front if self.zeta_front is not None else self.zeta[-1]
        z = np.linspace(self.zeta[0], hi, 200_001)
        return trapezoid(z**p.m * self(z), z), 1 / (p.i + 1)


# --------------------------------------------------------------------------
# Method 1: start at the front, integrate inward, rescale
# --------------------------------------------------------------------------
def _solve_front(p: Problem):
    n, m, b = p.n, p.m, p.beta
    z0 = 1.0 - FRONT_OFFSET  # front placed at zeta = 1
    f0 = (n * b * FRONT_OFFSET) ** (1 / n)  # local front form
    g0 = -(z0**m) * b * f0  # f^n f' = -beta zeta_N f
    z_end = 0.0 if m == 0 else ZETA_MIN_RADIAL

    sol = solve_ivp(
        p.rhs, (z0, z_end), [f0, g0], rtol=RTOL, atol=ATOL, dense_output=True
    )
    if not sol.success:
        raise RuntimeError(sol.message)
    F0, G0 = sol.y[:, -1]

    # Invariance f -> lam F, zeta -> mu zeta with mu = lam^(n/2).
    # Value scales as lam; flux scales as lam^(n+1) mu^(m-1).
    if p.bc == "dirichlet":
        lam = 1.0 / F0  # f(0) = 1
    else:
        lam = (-G0) ** (-1 / (n + 1 + n * (m - 1) / 2))  # flux = -1
    mu = lam ** (n / 2)

    def profile(z):
        out = np.zeros_like(z)
        inside = (z >= mu * z_end) & (z <= mu * z0)
        if inside.any():
            out[inside] = lam * sol.sol(z[inside] / mu)[0]
        return out

    return profile, mu, mu * z_end  # profile, zeta_front, first valid zeta


# --------------------------------------------------------------------------
# Method 2: shooting from the origin (no front)
# --------------------------------------------------------------------------
def _solve_shooting(p: Problem, zeta_max: float):
    m, n = p.m, p.n
    z_start = 0.0 if m == 0 else ZETA_MIN_RADIAL

    if p.bc == "dirichlet":  # f(0) = 1 known, flux g(0) = s unknown
        y0 = lambda s: [1.0, s]
        s0, positive = -1.0, False
    else:  # flux g = -1 known, f(origin) = s unknown
        y0 = lambda s: [s, -1.0]
        log_term = -np.log(z_start) if m else 1.0
        s0, positive = ((n + 1) * log_term) ** (1 / (n + 1)) + p.theta_inf, True

    def hits_zero(z, y):
        return y[0]

    hits_zero.terminal, hits_zero.direction = True, -1

    def shoot(s, dense=False):
        return solve_ivp(
            p.rhs,
            (z_start, zeta_max),
            y0(s),
            events=hits_zero,
            rtol=RTOL,
            atol=ATOL,
            dense_output=dense,
        )

    def miss(s):
        """f(zeta_max) - theta_inf; increasing in s, negative if f hits zero."""
        sol = shoot(s)
        if sol.status == 1:
            return (sol.t_events[0][0] - zeta_max) - p.theta_inf
        return sol.y[0, -1] - p.theta_inf

    lo, hi = _bracket(miss, s0, positive)
    s = brentq(miss, lo, hi, xtol=1e-13, rtol=1e-13)
    sol = shoot(s, dense=True)
    z_stop = sol.t[-1]

    def profile(z):
        out = np.full_like(z, np.nan if p.theta_inf else 0.0)
        ok = (z >= z_start) & (z <= z_stop)
        if ok.any():
            out[ok] = sol.sol(z[ok])[0]
        out[z > z_stop] = p.theta_inf if sol.status == 0 else 0.0
        return out

    return profile, z_start


def _bracket(miss, s0, positive, max_iter=60):
    """Find lo < hi with miss(lo) < 0 < miss(hi); miss is increasing."""
    lo = hi = s0
    step = abs(s0) if s0 else 1.0
    for _ in range(max_iter):
        if miss(lo) < 0:
            break
        lo = lo / 2 if positive else lo - step
        step *= 2
    step = abs(s0) if s0 else 1.0
    for _ in range(max_iter):
        if miss(hi) > 0:
            break
        hi = hi * 2 if positive else hi + step
        step *= 2
    if not (miss(lo) < 0 < miss(hi)):
        raise RuntimeError("could not bracket the shooting parameter")
    return lo, hi


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------
def solve(
    n: float,
    bc: BC,
    *,
    i: float = 0.0,
    geometry: Geometry = "planar",
    theta_inf: float = 0.0,
    zeta_max: float | None = None,
    n_points: int = 400,
) -> Solution:
    """Similarity solution; see the module docstring for the problem.

    zeta_max: end of the returned grid.  Defaults to the front position
    (front method) or ZETA_MAX_SHOOTING (shooting, also the far-field point).

    Example:  sol = solve(3, "neumann", i=1);  zeta, theta = sol
    """
    p = Problem(n=n, bc=bc, i=i, geometry=geometry, theta_inf=theta_inf)

    if p.has_front:
        profile, zeta_front, z_start = _solve_front(p)
        zmax = zeta_front if zeta_max is None else zeta_max
        method = "front"
    else:
        zmax = ZETA_MAX_SHOOTING if zeta_max is None else zeta_max
        profile, z_start = _solve_shooting(p, zmax)
        zeta_front, method = None, "shooting"

    zeta = np.linspace(z_start, zmax, n_points)
    return Solution(p, zeta, profile(zeta), zeta_front, method, profile)


# --------------------------------------------------------------------------
# Demo
# --------------------------------------------------------------------------
if __name__ == "__main__":
    cases = [
        {"n": 1, "bc": "dirichlet"},
        {"n": 3, "bc": "dirichlet"},
        {"n": 1, "bc": "neumann"},
        {"n": 3, "bc": "neumann"},
        {"n": 3, "bc": "neumann", "i": 1},  # ramp injection
        {"n": 3, "bc": "neumann", "i": 1, "geometry": "radial"},  # radial ramp
        {"n": 3, "bc": "dirichlet", "theta_inf": 0.2},  # no front
        {"n": 0, "bc": "dirichlet"},  # linear
    ]
    print(
        f"{'case':<52}{'alpha':>7}{'beta':>7}{'method':>10}{'zeta_N':>10}{'mass':>18}"
    )
    for c in cases:
        s = solve(**c)
        zN = f"{s.zeta_front:.5f}" if s.zeta_front else "-"
        mc = s.mass_check()
        mass = f"{mc[0]:.5f}/{mc[1]:.5f}" if mc else "-"
        label = ", ".join(f"{k}={v}" for k, v in c.items())
        print(
            f"{label:<52}{s.alpha:>7.3f}{s.beta:>7.3f}{s.method:>10}{zN:>10}{mass:>18}"
        )

    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 4.5))
        for c in cases[:6]:
            s = solve(**c, zeta_max=3.0)
            ax.plot(s.zeta, s.theta, label=", ".join(f"{k}={v}" for k, v in c.items()))
        ax.set(xlabel=r"$\zeta$", ylabel=r"$f(\zeta)$", ylim=(0, 2.5))
        ax.legend(fontsize=7)
        fig.savefig("similarity_profiles.png", dpi=150, bbox_inches="tight")
        print("\nplot saved to similarity_profiles.png")
    except ImportError:
        pass
