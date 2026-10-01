"""Analytical (n=0) and semi-analytical (n=3) similarity profiles for ramp injection.

Pure solver output -- theta(zeta) for the linearly-increasing injection rate
condition q(t) = m_q * t, with no numerical run involved. n=0 (rigid) is
linear and has a closed-form solution (fracinj.solvers.exact); n=3 (soft) is
nonlinear and needs the shooting-method solve (fracinj.solvers.similarity).
For validation against a numerical run, see long_fracture_v2.py, which
overlays these same two curves on aperture profiles from a halfspace run.
"""

import matplotlib.pyplot as plt
import numpy as np

from fracinj.solvers.exact import solve_linear_diffusion_ramp_flux
from fracinj.solvers.similarity import solve_neumann_ramp_n3

COLUMN_WIDTH = 3.4
GOLDEN_RATIO = (1 + 5**0.5) / 2

# %%

zeta_rigid = np.linspace(0, 4, 200)
theta_rigid = solve_linear_diffusion_ramp_flux(zeta_rigid)

zeta_soft, theta_soft = solve_neumann_ramp_n3(zeta_max=2.0)

# %%

fig = plt.figure(
    figsize=(COLUMN_WIDTH, COLUMN_WIDTH / GOLDEN_RATIO),
    dpi=150,
    layout="constrained",
    clear=True,
    num=1,
)
ax = fig.subplots()

ax.plot(zeta_rigid, theta_rigid, lw=1.5, label="Rigid ($n=0$), analytical")
ax.plot(zeta_soft, theta_soft, lw=1.5, label="Soft ($n=3$), semi-analytical")
ax.set(xlabel=r"$\zeta$", ylabel=r"$\theta$")
ax.legend()

fig.canvas.draw_idle()
plt.pause(0.01)

# fig.savefig("/home/kahmadov/phd/migration/figures/ramp-injection/similarity-profiles.eps")
# fig.savefig("/home/kahmadov/phd/migration/tex/2026-09-28/figures/similarity-profiles.eps")
