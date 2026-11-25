"""Schematic: sharp pressure front with a tensile normal-stress peak at the front."""

import matplotlib.pyplot as plt
import numpy as np

from fracinj import io, paths, physics, plotting
from fracinj.solvers.elastic import solve_elastic_joint

# %%
# --- data: pressure snapshot at mid-run, then the induced normal stress ---
run = io.read_halfspace(paths.results_dir("halfspace", "ramp", "m_q-1e-08.hdf5"))

scales = physics.ramp_rate_fixed_length_cubic(run.params)
x_c = scales.x_c
eps = physics.plane_strain_modulus(run.params) / (4 * run.params.k_n * x_c)

x = run.x_vert / x_c
p = run.p[len(run.t) // 2] / (scales.w_c * run.params.k_n)

# The solver needs a profile that decays at both ends, so solve on the
# mirrored domain and keep only x >= 0 for the plot.
X = np.linspace(-x[-1], x[-1], 2**12)
Pi = np.interp(np.abs(X), x, p)
Sigma, _ = solve_elastic_joint(X=X, Pi=Pi, eps=eps)

right = X >= 0
X, Pi, Sigma = X[right], Pi[right], Sigma[right]

# Schematic only: scale the stress so its tensile peak is half the pressure peak
Sigma *= 0.8 * Pi.max() / abs(Sigma.min())

i_front = np.argmin(Sigma)  # tensile peak marks the front
xf = X[i_front]
x_max = 2 * xf  # right edge of the plot, relative to the front

# %%
# --- schematic plot ---
# Included at 1:1 in overleaf/figures/model-schematic.tex, so text (10 pt)
# and line widths (0.8 pt = TikZ "thick") match the TikZ panel; the width is
# chosen so the whole schematic is 6.5 in wide (the column width).
W_FIG, H_FIG = 3.158, 2.24  # in
LW = 0.8  # pt
# serif Computer Modern, like the TikZ text (the "science" style sets sans-serif)
plt.rcParams.update({"font.family": "serif", "font.size": 10,
                     "text.latex.preamble": r"\usepackage{amsmath}"})
fig, ax = plt.subplots(figsize=(W_FIG, H_FIG), layout="constrained")

CLW = 1.5 * LW  # curve line width
inside = X <= xf  # p is zero beyond the front; drawing it there would thicken the x-axis
ax.plot(X[inside], Pi[inside], color="C0", lw=CLW)
ax.plot(X, Sigma, color="C1", lw=CLW)
ax.fill_between(X, Pi, 0, color="C0", alpha=0.25, lw=0)  # fluid-filled part of the fracture

# bare x-axis through y = 0, drawn as an arrow that runs past the plot edge
ax.axis("off")
x_end = 1.06  # arrow tip, in axes fractions (1 = right edge of the data)
ax.annotate("", xy=(x_end, 0), xytext=(0, 0), xycoords=ax.get_yaxis_transform(),
            arrowprops=dict(arrowstyle="-|>", lw=LW, color="k",
                            shrinkA=0, shrinkB=0, mutation_scale=8),
            zorder=3)  # draw above the curves
ax.set(
    xlim=(0, x_max),
    ylim=(1.5 * Sigma.min(), 1.2 * Pi.max()),
)
ax.text(x_end + 0.02, 0, r"$x$", transform=ax.get_yaxis_transform(), va="center")

# front line, and the injection point and front position labelled on the x-axis
ax.axvline(xf, ymin=0.08, ymax=0.88, color="k", ls=":", lw=1.5 * LW)  # axes fractions
ax.annotate(r"$0$", (0, 0), xytext=(0, -5), textcoords="offset points",
            ha="center", va="top")
ax.annotate(r"$\ell(t)$", (xf, 0), xytext=(-5, -3), textcoords="offset points",
            ha="right", va="top")
# label at the top of the front line: it migrates as a power law in time
ax.annotate("front\n" r"$\ell \propto t^{\beta}$", (xf, 0.88), xycoords=ax.get_xaxis_transform(),
            xytext=(4, 0), textcoords="offset points", ha="left", va="top")

# curve labels anchored to the data
i_p = np.searchsorted(X, 0.3 * xf)
ax.annotate(r"$p(x,t)$", (X[i_p], Pi[i_p]), xytext=(4, 4),
            textcoords="offset points", color="C0")
i_s = np.searchsorted(X, xf + 0.5 * (x_max - xf))
ax.annotate(r"$\Delta\sigma_{\mathrm{n}}(x,t)$", (X[i_s], Sigma[i_s]), xytext=(0, -6),
            textcoords="offset points", ha="center", va="top", color="C1")

# arrow to the tensile stress peak
ax.annotate("tensile peak", (xf, Sigma[i_front]), xytext=(25, -4), color="C1",
            textcoords="offset points", va="center",
            arrowprops=dict(arrowstyle="->", lw=LW, shrinkA=2, shrinkB=2, color="C1"))

# injection point at x = 0
ax.plot(0, 0, "o", color="C0", ms=4, clip_on=False, zorder=5)

fig.savefig(paths.OVERLEAF / "figures" / "schematic-front.pdf")
plt.show()
