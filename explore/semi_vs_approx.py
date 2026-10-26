import matplotlib.pyplot as plt
import numpy as np

from fracinj import plotting
from fracinj.solvers.similarity import solve_neumann_ramp_n3

# %% numerical solution
ze, th = solve_neumann_ramp_n3(zeta_max=0.7, n_points=500)

# %%
zf = ((5 / 7) ** (1 / 5) * 2 ** (3 / 5)) / 3 ** (4 / 5)

th_app = lambda x: ((21 / 5) * zf * (zf - x))**(1/3)
ze_app = np.linspace(0, zf, 200)

# %%
CW, GR = 3.4, 1.618
fig = plt.figure(
    figsize=(CW, CW / GR), dpi=150, layout="constrained", clear=True, num=1
)
ax = fig.subplots()
ax.plot(ze, th, label="Semi-analytical")
ax.plot(
    ze_app,
    th_app(ze_app),
    ls="--",
    label=r"$\theta \approx \left[ \frac{21}{5} \zeta_f (\zeta_f - \zeta) \right]^{1/3}$",
)
ax.legend(loc="lower left")
ax.text(
    0.97,
    0.95,
    rf"$\zeta_f = \dfrac{{(5/7)^{{1/5}} \, 2^{{3/5}}}}{{3^{{4/5}}}} \approx {zf:.3f}$",
    transform=ax.transAxes,
    ha="right",
    va="top",
)
ax.set(
    xlabel=r"$\zeta$",
    ylabel=r"$\theta$",
    ylim=(-0.05, 1.5),
    title="Ramp injection, $n=3$",
)
fig.canvas.draw_idle()
plt.pause(0.01)

# %%

for ext in ("eps", "png"):
    plotting.save_figure(fig, "similarity-ramp-n3-approx", ext=ext)
