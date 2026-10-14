"""Checks of gc_slip.py quoted in ../main.tex (Sec. 'Verification').

Run from anywhere:  python tex/2026-10-14_gauss-chebyshev/code/verify.py
"""

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import erfc, eval_chebyu, i0e

from gc_slip import coefficient_matrix, hilbert_matrix, nodes, solve_slip


def p_erfc(xi):
    """Constant injection pressure: xi in units of L = sqrt(4 alpha t)."""
    return erfc(np.abs(xi))


def p_gauss(xi):
    """Smooth (analytic) test profile."""
    return np.exp(-(xi**2))


def lam_erfc(T):
    """Continuous K = 0 condition for p_erfc, by adaptive quadrature."""
    g = lambda lam: quad(lambda th: erfc(lam * abs(np.cos(th))), 0, np.pi,
                         points=[np.pi / 2], limit=400)[0] / np.pi - T
    return brentq(g, 1e-10, 1e4, xtol=1e-14)


def lam_gauss(T):
    """Continuous K = 0 condition for p_gauss: T = exp(-lam^2/2) I0(lam^2/2)."""
    return brentq(lambda lam: i0e(lam**2 / 2) - T, 1e-10, 1e4, xtol=1e-14)


if __name__ == "__main__":
    n = 16
    s, w, x = nodes(n)
    H = hilbert_matrix(s, w, x)

    print("1) singular quadrature, F = U_m:  max_i |H U_m(s) - T_{m+1}(x_i)|   (n = 16)")
    for m in (0, 1, n - 1, n, 2 * n, 2 * n + 1):
        err = np.max(np.abs(H @ eval_chebyu(m, s) - np.cos((m + 1) * np.arccos(x))))
        print(f"   m = {m:3d}: {err:.1e}")

    print(f"2) column sums of H:  max_j |sum_i H_ij| = {np.max(np.abs(H.sum(axis=0))):.1e}")

    print("3) off-crack transform:  (1/pi) int sqrt(1-s^2) U_k(s) / (s - x) ds  vs  -rho^(k+1)")
    for xo in (1.3, -2.0):
        rho = xo - np.sign(xo) * np.sqrt(xo**2 - 1)
        # weight='alg' integrates f(t) (1 + t)^(1/2) (1 - t)^(1/2) exactly at the ends
        errs = [abs(quad(lambda t: eval_chebyu(k, t) / (t - xo), -1, 1, weight="alg", wvar=(0.5, 0.5),
                         epsabs=1e-13, epsrel=1e-12)[0] / np.pi + rho ** (k + 1))
                for k in range(6)]
        print(f"   x = {xo:+.1f}: max over k = 0..5: {max(errs):.1e}")

    print("4) manufactured: T = 0, p = 3/2 - 3 y^2  ->  a = 1, delta = (1 - x^2)^(3/2)")
    xq = np.linspace(-1, 1, 201)
    xo = np.array([1.01, 1.5, 3.0, -2.0])
    rho = xo - np.sign(xo) * np.sqrt(xo**2 - 1)
    for n in (2, 4, 32):
        sol = solve_slip(lambda y: 1.5 - 3 * y**2, 0.0, n=n, bracket=(0.1, 10))
        print(f"   n = {n:2d}: |a - 1| = {abs(sol.a - 1):.0e},"
              f"  max|delta - exact| = {np.max(np.abs(sol.slip(xq) - (1 - xq**2) ** 1.5)):.0e},"
              f"  max|dtau - (3x^2 - 3/2)| = {np.max(np.abs(sol.stress(xq) - (3 * xq**2 - 1.5))):.0e},"
              f"  off-crack |dtau - 3/2 rho^2| = {np.max(np.abs(sol.stress(xo) - 1.5 * rho**2)):.0e}")

    print("5) crack half-length lambda = a/L for p = erfc:  rel. error vs the continuous condition")
    for T in (0.02, 0.1, 0.5, 0.9, 0.99):
        lr = lam_erfc(T)
        errs = []
        for n in (32, 128, 512):
            try:
                errs.append(f"{abs(solve_slip(p_erfc, T, n=n).a - lr) / lr:.1e}")
            except ValueError:  # no sign change: pressure zone not resolved
                errs.append("  --   ")
        print(f"   T = {T:4.2f}: lambda = {lr:8.4f}  [2/(pi^1.5 T) = {2 / (np.pi**1.5 * T):8.4f},"
              f" pi^1.5 (1-T)/4 = {np.pi**1.5 * (1 - T) / 4:6.4f}]   n = 32, 128, 512: {' '.join(errs)}")

    print("6) convergence of delta(0) at T = 0.5 (reference n = 4001)")
    for name, p, lam in (("erfc ", p_erfc, lam_erfc), ("gauss", p_gauss, lam_gauss)):
        ref = solve_slip(p, 0.5, n=4001)
        d_ref = ref.slip(0.0)[0]
        line = []
        for n in (8, 16, 32, 64, 128):
            sol = solve_slip(p, 0.5, n=n)
            line.append(f"{abs(sol.slip(0.0)[0] - d_ref):.1e}")
        print(f"   {name}: lambda = {ref.a:.10f} (exact {lam(0.5):.10f}), delta(0) = {d_ref:.8f};"
              f"  err at n = 8..128: {' '.join(line)}")
        print(f"          lstsq residual = {ref.residual:.1e},"
              f"  delta(1) = {ref.slip(1.0)[0]:.1e},  A_0 = {(coefficient_matrix(ref.s) @ ref.F)[0]:.1e}")
