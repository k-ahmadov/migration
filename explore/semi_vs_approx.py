import matplotlib.pyplot as plt
import numpy as np

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
ax.plot(ze, th)
ax.plot(ze_app, th_app(ze_app))
ax.set(
    xlabel=r"$\zeta$",
    ylabel=r"$\theta$",
    title="Ramp injection, $n=3$",
)
fig.canvas.draw_idle()
plt.pause(0.01)
