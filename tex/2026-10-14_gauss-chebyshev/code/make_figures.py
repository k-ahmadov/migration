"""Figures of ../main.tex, written to ../figures/.

Run from anywhere:  python tex/2026-10-14_gauss-chebyshev/code/make_figures.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from gc_slip import crack_length, nodes, solve_slip
from verify import lam_erfc, p_erfc, p_gauss

plt.style.use(["science", "bright"])
FIGURES = Path(__file__).resolve().parents[1] / "figures"
SIZE = (2.7, 2.1)


def crack_length_figure():
    T = np.linspace(0.005, 0.995, 100)
    _, _, x = nodes(4096)
    lam = [crack_length(p_erfc, Ti, x) for Ti in T]
    T_ref = np.array([0.02, 0.1, 0.3, 0.5, 0.7, 0.9, 0.98])
    fig, ax = plt.subplots(figsize=SIZE)
    ax.semilogy(T, lam, "k-", label=r"Gauss--Chebyshev, $n=4096$")
    ax.semilogy(
        T_ref,
        [lam_erfc(Ti) for Ti in T_ref],
        "ko",
        ms=3,
        mfc="none",
        label="adaptive quadrature",
    )
    ax.semilogy(T, 2 / (np.pi**1.5 * T), "--", label=r"$2/(\pi^{3/2}T)$")
    ax.semilogy(T, np.pi**1.5 * (1 - T) / 4, ":", label=r"$\pi^{3/2}(1-T)/4$")
    ax.set(
        xlabel=r"$T$",
        ylabel=r"$\lambda = a/\sqrt{4\alpha t}$",
        xlim=(0, 1),
        ylim=(5e-3, 2e2),
    )
    ax.legend(fontsize=6)
    fig.savefig(FIGURES / "crack-length.pdf")


def slip_profile_figure():
    fig, ax = plt.subplots(figsize=SIZE)
    for T in (0.1, 0.3, 0.5, 0.7, 0.9):
        sol = solve_slip(p_erfc, T, n=512)
        xq = np.linspace(-1, 1, 401)
        ax.plot(sol.a * xq, sol.a * sol.slip(xq), label=rf"$T={T}$")
    ax.set(
        xlabel=r"$x/\sqrt{4\alpha t}$",
        ylabel=r"$\delta\, G'/(f\Delta p\sqrt{4\alpha t})$",
        xlim=(-4, 4),
        ylim=(0, None),
    )
    ax.legend(fontsize=6)
    fig.savefig(FIGURES / "slip-profiles.pdf")


def stress_figure():
    fig, ax = plt.subplots(figsize=SIZE)
    xq = np.linspace(-3, 3, 1201)
    for T, color in ((0.1, "C0"), (0.5, "C2")):
        sol = solve_slip(p_erfc, T, n=512)
        ax.plot(xq, T - p_erfc(sol.a * xq), "--", color="0.6", lw=1.6)
        ax.plot(xq, sol.stress(xq), color=color, label=rf"$T={T}$")
    ax.plot([], [], "--", color="0.6", lw=1.6, label=r"$T - p(ax)$")
    ax.axvspan(-1, 1, color="0.92", zorder=0)
    ax.set(xlabel=r"$x/a$", ylabel=r"$\Delta\tau / (f\Delta p)$", xlim=(-3, 3))
    ax.legend(fontsize=6, loc="lower left")
    fig.savefig(FIGURES / "stress.pdf")


def convergence_figure():
    ns = 2 ** np.arange(2, 10)
    fig, ax = plt.subplots(figsize=SIZE)
    for p, label, marker in (
        (p_erfc, r"$p=\mathrm{erfc}|\xi|$", "o"),
        (p_gauss, r"$p=e^{-\xi^2}$", "s"),
    ):
        d_ref = solve_slip(p, 0.5, n=4001).slip(0.0)[0]
        err = [abs(solve_slip(p, 0.5, n=n).slip(0.0)[0] - d_ref) for n in ns]
        ax.loglog(ns, np.maximum(err, 1e-16), marker + "-", ms=3, label=label)
    ax.loglog(ns, 0.5 * ns**-2.0, "k:", label=r"$\propto n^{-2}$")
    ax.set(xlabel=r"$n$", ylabel=r"error in $\delta(0)$", ylim=(1e-16, 1))
    ax.legend(fontsize=6)
    fig.savefig(FIGURES / "convergence.pdf")


if __name__ == "__main__":
    FIGURES.mkdir(exist_ok=True)
    crack_length_figure()
    slip_profile_figure()
    stress_figure()
    convergence_figure()
