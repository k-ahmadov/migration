import numpy as np
import matplotlib.pyplot as plt

from fracinj.solvers import slip

# %%
L = 50.0
l = 5.0
p0 = 0.0
dp = 8e6
tau0 = 3e6
sigma0 = 10e6
f = np.tan(30 * np.pi / 180)
E = 60e9
nu = 0.25
G = E / (2 * (1 + nu))
G_ = G / (2 * (1 - nu))
sigma_eff0 = sigma0 - p0
T = (1 - tau0 / (f * sigma_eff0)) / (dp / sigma_eff0)


# %%
p_f = lambda x: np.heaviside(l - np.abs(x), 0.5)
sol = slip.solve_slip(p=p_f, T=T)

# %%
CW, GR = 3.4, 1.618
fig = plt.figure(figsize=(CW, CW/GR), dpi=150, layout='constrained', clear=True, num=1)
ax = fig.subplots()
for T in (0.1, 0.3, 0.5, 0.7, 0.9):
    sol = slip.solve_slip(p=p_f, T=T)
    ax.plot(sol.a * sol.s / l, sol.a * sol.delta / l, label=rf"$T={T}$")
ax.set(
    xlabel=r"$x / l$",
    ylabel=r"$\delta\, G'/(f\Delta p l)$",
    # xlim=(-4, 4),
    ylim=(0, None),
)
ax.legend()
fig.canvas.draw_idle()
plt.pause(0.01)
