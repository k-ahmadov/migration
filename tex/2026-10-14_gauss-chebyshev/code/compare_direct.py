"""Direct scheme (explore/slip/gc.py) vs collocation (gc_slip.py), quoted in
../main.tex, Sec. 'A direct scheme for a given pressure'.

Run from anywhere:  python tex/2026-10-14_gauss-chebyshev/code/compare_direct.py
"""

import importlib.util
import time
from pathlib import Path

import numpy as np
from scipy.special import eval_chebyt, eval_chebyu

from gc_slip import solve_slip
from verify import p_erfc, p_gauss

# 'gc' is the name of a built-in module, so explore/slip/gc.py is loaded by path
GC = Path(__file__).resolve().parents[3] / "explore" / "slip" / "gc.py"
spec = importlib.util.spec_from_file_location("gc_direct", GC)
direct = importlib.util.module_from_spec(spec)
spec.loader.exec_module(direct)

L_PATCH, T_PATCH = 5.0, 0.6


def p_patch(xi):
    """Uniform pressure patch of half-width L_PATCH."""
    return np.heaviside(L_PATCH - np.abs(xi), 0.5)


def slip_patch(x, l):
    """Exact slip for the patch, l = L_PATCH / a (from explore/slip/analysis/patch_slip.py)."""
    L, X = np.sqrt(1 - l**2), np.sqrt(np.clip(1 - x**2, 0, None))
    with np.errstate(divide="ignore", invalid="ignore"):
        h = (x * np.log(np.abs((x * L - l * X) / (x * L + l * X)))
             + l * np.log(np.abs((L + X) / (L - X)))) / np.pi
    return np.where(np.isfinite(h), h, 2 * l / np.pi * np.log(1 / l))


if __name__ == "__main__":
    print("1) first-kind singular rule, g = T_m:  max_i |(1/N) sum_j T_m(s_j)/(x_i - s_j) + U_{m-1}(x_i)|")
    N = 16
    s, x = direct.nodes(N)  # s: roots of T_N, x: roots of U_{N-1}
    for m in (1, N, 2 * N, 2 * N + 1):
        quad = 1 / N * np.sum(eval_chebyt(m, s) / (x[:, None] - s), axis=1)
        print(f"   N = {N}, m = {m:2d}: {np.max(np.abs(quad + eval_chebyu(m - 1, x))):.1e}")

    print("2) same discrete slip gradient: direct phi vs collocation F (N = n + 1 nodes)")
    for name, p, T in (("erfc", p_erfc, 0.5), ("gauss", p_gauss, 0.5), ("patch", p_patch, T_PATCH)):
        for N in (17, 257):
            d, c = direct.solve_slip(p, T, n=N), solve_slip(p, T, n=N - 1)
            F = np.interp(d.x, c.s[::-1], c.F[::-1])  # same points, reordered
            print(f"   {name:5s} N = {N:3d}: |a_d - a_c| = {abs(d.a - c.a):.0e},"
                  f"  max|phi - F| = {np.max(np.abs(d.phi - F)):.0e}")

    print("3) max slip error at the first-kind nodes")
    refs = {"gauss": solve_slip(p_gauss, 0.5, n=8000), "erfc": solve_slip(p_erfc, 0.5, n=8000)}
    d = direct.solve_slip(p_erfc, 0.5, n=8001)
    print(f"   reference check, erfc: max|direct(N = 8001) - collocation(n = 8000)| ="
          f" {np.max(np.abs(d.delta - refs['erfc'].slip(d.s))):.0e}")
    a_patch = L_PATCH / np.sin(np.pi * T_PATCH / 2)
    cases = (
        ("patch", p_patch, T_PATCH, lambda s: slip_patch(s, L_PATCH / a_patch)),
        ("gauss", p_gauss, 0.5, lambda s: refs["gauss"].slip(s)),
        ("erfc", p_erfc, 0.5, lambda s: refs["erfc"].slip(s)),
    )
    for name, p, T, exact in cases:
        for N in (16, 64, 256, 1024):
            d, c = direct.solve_slip(p, T, n=N), solve_slip(p, T, n=N - 1)
            ex = exact(d.s)
            print(f"   {name:5s} N = {N:4d}:  direct {np.max(np.abs(d.delta - ex)):.1e}"
                  f"   collocation {np.max(np.abs(c.slip(d.s) - ex)):.1e}")

    print("4) run time, p = erfc, T = 0.5")
    for N in (1024, 4096):
        t0 = time.perf_counter()
        direct.solve_slip(p_erfc, 0.5, n=N)
        t1 = time.perf_counter()
        solve_slip(p_erfc, 0.5, n=N - 1)
        t2 = time.perf_counter()
        print(f"   N = {N:4d}: direct {t1 - t0:.3f} s   collocation {t2 - t1:.3f} s")
