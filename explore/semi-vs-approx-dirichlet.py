import matplotlib.pyplot as plt
import numpy as np

from fracinj.solvers.similarity import solve_similarity

# %%

zeta, theta = solve_similarity(n=3, bc="dirichlet", theta_inf=0.0, zeta_max=1.0)

# %%
x_f = np.sqrt(2 / 3)
x = np.linspace(0, 1)


def th_ap(x):
    return (1 - x / x_f) ** (1 / 3)


# %%

fig, ax = plt.subplots(clear=True, num=1, dpi=200, layout="constrained")
ax.plot(zeta, theta)
ax.plot(x, th_ap(x))

fig.canvas.draw_idle()
plt.pause(0.01)
