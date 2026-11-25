"""Closed-form and approximate similarity solutions of diffusion problems.

Conventions are the same as ``similarity.py`` (dimensionless; for a
diffusivity D use x / sqrt(D) or tau = D t):

    (1/x^m) d/dx [ x^m w^n dw/dx ] = dw/dt,      m = 0 planar, m = 1 radial

    "dirichlet":   w(0, t) - w0 = t^i
    "neumann":     -x^m w^n dw/dx -> t^i  as x -> 0     (injection flux)

    w = w0 + t^alpha f(zeta),   zeta = x / t^beta

Exact solutions (linear, n = 0; beta = 1/2, eta = zeta/2, s = zeta^2/4)
----------------------------------------------------------------------
    planar, dirichlet:  f = Gamma(i+1) 4^i       i^{2i}   erfc(eta),  alpha = i
    planar, neumann:    f = Gamma(i+1) 2^{2i+1}  i^{2i+1} erfc(eta),  alpha = i + 1/2
    radial, neumann:    f = Gamma(i+1)/2 e^{-s} U(i+1, 1, s),         alpha = i

where i^k erfc is the k-th repeated integral of erfc and U is Tricomi's
confluent hypergeometric function.  These hold for any real i >= 0, e.g.
    i = 0:  erfc,  2 ierfc,  E1(s)/2           (Theis)
    i = 1:  4 i^2erfc,  8 i^3erfc,  [(1+s) E1(s) - e^{-s}]/2

Rigid stress front (linear, n = 0)
---------------------------------
    rigid_stress_front: position zeta_f = x_f / sqrt(D t) of the most tensile
                      point of the normal stress induced by the exact linear
                      pressure, in the long-pulse limit of the elastic joint
                      (sigma = |k| p in Fourier/Hankel space). The profile is
                      self-similar, so x_f = zeta_f sqrt(D t) exactly; e.g.
                      planar ramp zeta_f = 1.3434 (eta_f = zeta_f / 2 = 0.6717).

Approximate solutions
---------------------
    radial_borehole:  linear, aperture held on a borehole wall (late-time
                      Jacob-Lohman approximation; no similarity solution
                      exists for a fixed radius).
    front_approximation: nonlinear (n > 0), front-form profile
                      f = [n beta zeta_N (zeta_N - zeta)]^(1/n) with zeta_N
                      set by mass balance (Neumann) or f(0) = 1 (Dirichlet).

The old function names are kept at the bottom as thin wrappers with their
original variables and normalisations.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.integrate import quad
from scipy.optimize import minimize_scalar
from scipy.special import beta as beta_fn
from scipy.special import ellipk, exp1, gamma, hyperu, pbdv

BC = Literal["dirichlet", "neumann"]
Geometry = Literal["planar", "radial"]
Array = NDArray[np.float64]


# --------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------
def exponents(n: float, bc: BC, i: float = 0.0, geometry: Geometry = "planar"):
    """Similarity exponents (alpha, beta); identical to similarity.Problem."""
    m = 0 if geometry == "planar" else 1
    alpha = i if bc == "dirichlet" else (2 * i + 1 - m) / (n * (m + 1) + 2)
    return alpha, (n * alpha + 1) / 2


def ierfc(k: float, z: ArrayLike) -> Array:
    """k-th repeated integral of erfc, i^k erfc(z), for real k > -1.

    Uses i^k erfc(z) = exp(-z^2/2) D_{-k-1}(sqrt(2) z) / sqrt(2^(k-1) pi),
    which is stable for large z (unlike the usual recurrence).
    """
    z = np.asarray(z, dtype=float)
    d, _ = pbdv(-k - 1, np.sqrt(2) * z)
    return np.exp(-z * z / 2) * d / np.sqrt(2 ** (k - 1) * np.pi)


def _check(bc: str, geometry: str):
    if bc not in ("dirichlet", "neumann"):
        raise ValueError(f"unknown boundary condition {bc!r}")
    if geometry not in ("planar", "radial"):
        raise ValueError(f"unknown geometry {geometry!r}")
    if bc == "dirichlet" and geometry == "radial":
        raise ValueError("radial Dirichlet has no similarity solution; use radial_borehole")


# --------------------------------------------------------------------------
# Exact solutions: linear diffusion (n = 0)
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class LinearSolution:
    bc: BC
    i: float = 0.0
    geometry: Geometry = "planar"

    def __post_init__(self):
        _check(self.bc, self.geometry)

    @property
    def alpha(self) -> float:
        return exponents(0, self.bc, self.i, self.geometry)[0]

    beta = 0.5

    def __call__(self, zeta: ArrayLike) -> Array:
        """Profile f(zeta), zeta = x / sqrt(t), for a unit-strength boundary."""
        zeta = np.asarray(zeta, dtype=float)
        i = self.i
        if self.geometry == "radial":
            s = zeta**2 / 4
            return gamma(i + 1) / 2 * np.exp(-s) * hyperu(i + 1, 1, s)
        if self.bc == "dirichlet":
            return gamma(i + 1) * 4**i * ierfc(2 * i, zeta / 2)
        return gamma(i + 1) * 2 ** (2 * i + 1) * ierfc(2 * i + 1, zeta / 2)

    def w(self, x: ArrayLike, t: float, *, w0: float = 0.0, strength: float = 1.0) -> Array:
        """w(x, t) = w0 + strength * t^alpha f(x / sqrt(t))."""
        return w0 + strength * t**self.alpha * self(np.asarray(x, dtype=float) / np.sqrt(t))


def exact(bc: BC, *, i: float = 0.0, geometry: Geometry = "planar") -> LinearSolution:
    """Exact linear similarity solution.  Example: exact("neumann", i=1)(zeta)."""
    return LinearSolution(bc, i, geometry)


# --------------------------------------------------------------------------
# Rigid stress front: most tensile point of sigma = |k| p for the exact p
# --------------------------------------------------------------------------
_Z_MAX = 40.0  # integration cutoff in zeta; the profiles decay like exp(-zeta^2/4)


def _planar_stress(x: float, k: float) -> float:
    """H[dp/dzeta](x) up to a positive factor, for p = i^k erfc(|zeta|/2).

    dp/dzeta = -sign(zeta) g(|zeta|)/2 with g = i^(k-1) erfc(zeta/2), and the
    Hilbert transform (1/pi) PV int p'(s)/(x - s) ds of this odd function is
    PV int_0^inf g/(s - x) ds + int_0^inf g/(s + x) ds  (times 1/(2 pi)).
    """
    def g(s):
        return ierfc(k - 1, s / 2)

    pv = quad(g, 0, _Z_MAX, weight="cauchy", wvar=x, limit=200)[0]
    regular = quad(lambda s: g(s) / (s + x), 0, _Z_MAX, limit=200)[0]
    return pv + regular


def _radial_stress(r: float, i: float) -> float:
    """(-Laplacian)^(1/2) p at radius r, radial Neumann source of strength t^i.

    Uses (-Lap)^(1/2) p = (-Lap)^(-1/2) (-Lap p), with -Lap p = S delta - p_t
    (the diffusion equation, D = 1, t = 1) and the 2D kernel 1/(2 pi |x|),
    whose ring average is 4 K(m) / (r + rho), m = 4 r rho / (r + rho)^2.
    p_t is the response to the source derivative i t^(i-1): i p_{i-1}
    (for i = 0, the Theis p_t = exp(-zeta^2/4) / 2).
    """
    prev = None if i == 0 else exact("neumann", i=i - 1, geometry="radial")

    def p_t(z):
        return np.exp(-z * z / 4) / 2 if prev is None else i * prev(z)

    source = 2 * np.pi * quad(lambda z: p_t(z) * z, 0, _Z_MAX, limit=200)[0]

    def ring(z):
        return p_t(z) * z * 4 * ellipk(4 * r * z / (r + z) ** 2) / (r + z)

    # log singularity of K at rho = r: integrate either side of it
    spread = quad(ring, 0, r, limit=200)[0] + quad(ring, r, _Z_MAX, limit=200)[0]
    return (source / r - spread) / (2 * np.pi)


@functools.cache
def rigid_stress_front(
    bc: BC = "neumann", i: float = 0.0, geometry: Geometry = "planar"
) -> float:
    """zeta_f = x_f / sqrt(D t) of the rigid-regime stress front.

    The front is the most tensile point (``argmin`` of the normal stress, as in
    ``detection.find_stress_front``) of the stress induced by the exact linear
    pressure ``exact(bc, i=i, geometry=geometry)``, in the long-pulse limit of
    the elastic joint (L_k = E'/(4 k_n) << sqrt(D t)), where
    ``fracinj.solvers.elastic`` reduces to sigma = eps |k| p. That operator has
    no length scale, so the stress profile is self-similar in zeta and

        x_f = zeta_f sqrt(D t)        (eta_f = x_f / (2 sqrt(D t)) = zeta_f / 2).

    Values: planar ramp (neumann, i=1) 1.3434; planar constant rate 1.8483;
    radial ramp 1.6104; radial constant rate 2.1983. A finite L_k / sqrt(D t)
    pushes the front out (by ~3% at L_k / sqrt(D t) = 0.03 for the planar ramp).
    """
    _check(bc, geometry)
    if geometry == "planar":
        k = 2 * i if bc == "dirichlet" else 2 * i + 1  # p = i^k erfc(|zeta|/2)

        def stress(x):
            return _planar_stress(x, k)
    else:
        def stress(x):
            return _radial_stress(x, i)

    # coarse scan for the minimum (the radial stress is singular at 0), then refine
    grid = np.linspace(0.05, 6.0, 120)
    j = int(np.argmin([stress(x) for x in grid]))
    lo, hi = grid[max(j - 1, 0)], grid[min(j + 1, grid.size - 1)]
    res = minimize_scalar(stress, bounds=(lo, hi), method="bounded",
                          options={"xatol": 1e-9})
    return float(res.x)


# --------------------------------------------------------------------------
# Approximate solutions
# --------------------------------------------------------------------------
def radial_borehole(zeta: ArrayLike, zeta_b: float, theta_inf: float) -> Array:
    """Linear radial diffusion, aperture held at the borehole wall (approximate).

    theta = w / w_b, theta_inf = w0 / w_b, zeta = r / sqrt(t):

        theta = 1 + (theta_inf - 1) (1 - E1(zeta^2/4) / E1(zeta_b^2/4)).

    Exact for a boundary on the expanding circle zeta = zeta_b.  For a fixed
    borehole radius r_b, evaluating it with zeta_b = r_b / sqrt(t) is the
    late-time (Jacob-Lohman) approximation, valid for zeta_b << 1; the error
    decays only like 1/ln(1/zeta_b) (~0.15, 0.06, 0.03 |theta_inf - 1| at
    zeta_b = 1, 0.32, 0.1 against a fixed-r_b numerical solution).
    """
    zeta = np.asarray(zeta, dtype=float)
    return 1 + (theta_inf - 1) * (1 - exp1(zeta**2 / 4) / exp1(zeta_b**2 / 4))


@dataclass(frozen=True)
class FrontApproximation:
    n: float
    bc: BC
    i: float = 0.0
    geometry: Geometry = "planar"

    def __post_init__(self):
        _check(self.bc, self.geometry)
        if self.n <= 0:
            raise ValueError("front approximation needs n > 0")

    @property
    def alpha(self) -> float:
        return exponents(self.n, self.bc, self.i, self.geometry)[0]

    @property
    def beta(self) -> float:
        return exponents(self.n, self.bc, self.i, self.geometry)[1]

    @property
    def zeta_front(self) -> float:
        n, b = self.n, self.beta
        if self.bc == "dirichlet":                    # f(0) = 1
            return (n * b) ** -0.5
        # Neumann: int_0^zN zeta^m f dzeta = 1/(i+1)  (exact mass balance)
        m = 0 if self.geometry == "planar" else 1
        integral = (n * b) ** (1 / n) * beta_fn(m + 1, 1 + 1 / n)
        return (1 / ((self.i + 1) * integral)) ** (1 / (m + 1 + 2 / n))

    def __call__(self, zeta: ArrayLike) -> Array:
        """Approximate profile f(zeta); symmetric in zeta, zero beyond the front."""
        zeta = np.asarray(zeta, dtype=float)
        zN = self.zeta_front
        return (self.n * self.beta * zN * np.clip(zN - np.abs(zeta), 0, None)) ** (1 / self.n)

    def w(self, x: ArrayLike, t: float) -> Array:
        return t**self.alpha * self(np.asarray(x, dtype=float) / t**self.beta)


def front_approximation(
    n: float, bc: BC, *, i: float = 0.0, geometry: Geometry = "planar"
) -> FrontApproximation:
    """Front-form approximation; e.g. front_approximation(3, "neumann", i=1)."""
    return FrontApproximation(n, bc, i, geometry)


# --------------------------------------------------------------------------
# Backward-compatible wrappers (original names, variables and scalings)
# --------------------------------------------------------------------------
def solve_linear_diffusion_const_flux(zeta: Array) -> Array:
    """1D, constant flux; zeta = x / sqrt(D t).  Same as exact("neumann")."""
    return exact("neumann")(zeta)


def solve_linear_diffusion_ramp_flux(eta: Array) -> Array:
    """1D, ramp flux; f = 8 i^3erfc(eta), eta = x / (2 sqrt(D t)), f'(0) = -2."""
    return exact("neumann", i=1)(2 * np.asarray(eta, dtype=float))


def solve_linear_radial_diffusion(theta_inf: float, zeta_b: float, zeta: Array) -> Array:
    """Borehole approximation; zeta, zeta_b = r / (2 sqrt(D t))."""
    return radial_borehole(2 * np.asarray(zeta, dtype=float), 2 * zeta_b, theta_inf)


def solve_linear_radial_diffusion_const_flux(zeta: Array) -> Array:
    """Theis, f = E1(zeta^2), zeta = r / (2 sqrt(D t)); (w - w0) = Q/(4 pi D) f."""
    return 2 * exact("neumann", geometry="radial")(2 * np.asarray(zeta, dtype=float))


def solve_linear_radial_diffusion_ramp_flux(zeta: Array) -> Array:
    """Ramp point source, f = (1 + z) E1(z) - e^{-z}, z = zeta^2, zeta = r/(2 sqrt(D t))."""
    return 2 * exact("neumann", i=1, geometry="radial")(2 * np.asarray(zeta, dtype=float))


def solve_nonlinear_diffusion_ramp_flux(n: int = 2**10, extend: float = 1.0):
    """n = 3, ramp flux, front approximation on a symmetric grid (n = number of points).

    Returns (zeta_front, zeta, theta).
    """
    approx = front_approximation(3, "neumann", i=1)
    zf = approx.zeta_front
    zeta = np.linspace(-extend * zf, extend * zf, int(n * extend))
    return zf, zeta, approx(zeta)


# --------------------------------------------------------------------------
# Demo
# --------------------------------------------------------------------------
if __name__ == "__main__":
    z = np.array([0.0, 0.5, 1.0, 2.0, 4.0])
    print("exact linear profiles f(zeta), zeta = x/sqrt(t)")
    for bc, g in [("dirichlet", "planar"), ("neumann", "planar"), ("neumann", "radial")]:
        for i in (0, 1):
            sol = exact(bc, i=i, geometry=g)
            zz = z if g == "planar" else z[1:]
            vals = "  ".join(f"{v:8.5f}" for v in sol(zz))
            print(f"  {bc:<9} {g:<6} i={i}  alpha={sol.alpha:<4}  {vals}")

    print("\nfront approximation (nonlinear): zeta_N")
    for n, bc, i in [(1, "dirichlet", 0), (3, "dirichlet", 0), (1, "neumann", 0),
                     (3, "neumann", 0), (3, "neumann", 1)]:
        print(f"  n={n} {bc:<9} i={i}:  {front_approximation(n, bc, i=i).zeta_front:.5f}")
