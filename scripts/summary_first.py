import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from fracinj import io, paths, physics, plotting
from fracinj.solvers.elastic import FIE_log_sing
from fracinj.solvers.similarity import solve_dirichlet_n3
from fracinj.types import Parameters

# %%

params = Parameters(E=60e9, L=100.0, k_n=30e9, mu=1e-3, nu=0.25, w_i=1e-4, DP=9e6)
t = 10

# %% pressure

D = physics.diffusivity(params)
w0 = params.w_i + params.DP / params.k_n
theta_inf = params.w_i / w0
zeta, theta = solve_dirichlet_n3(theta_inf=theta_inf, zeta_max=2)
x = zeta * np.sqrt(D * t)
w = theta * w0
p = params.k_n * (w - params.w_i)
x_front = x[np.argmax(p <= 1e-3 * params.DP)]  # first point where p has vanished

# %% stress
E_plane = physics.plane_strain_modulus(params)
lam = (4.0 / np.pi) * params.k_n * params.L / E_plane
sn_char = params.DP * E_plane / (params.k_n * params.L)
x_mirror = np.concatenate((-x[:0:-1], x))  # avoid double-counting x=0
p_mirror = np.concatenate((p[:0:-1], p)) / sn_char
p_f = lambda s: np.interp(s * params.L, x_mirror, p_mirror)
x_hat, sn_hat, _ = FIE_log_sing(lam, p_f, 256)

# %% 3DEC run with the same parameters, for comparison

run = io.read_pickle(paths.results_dir("3dec", "constant-pressure") / "soft.pkl")
it = np.argmin(np.abs(run.t - t))
x_3dec, sn_3dec = io.sort_fields(run.x_sc, run.sn)
sn_3dec = sn_3dec[it]
x_3dec_p, p_3dec = io.sort_fields(run.x_sc, run.p)
p_3dec = p_3dec[it]

# %%

CW, GR = 3.4, 1.618
fig = plt.figure(
    figsize=(CW, CW / GR), dpi=200, layout="constrained", clear=True, num=1
)
ax = fig.subplots()
ax.axvspan(0.8 * x_front, 1.2 * x_front, color="0.85", lw=0, zorder=0)
ax.plot(x_3dec_p, p_3dec / 1e6, ".", markevery=0.1, color="C0")
ax.plot(x, p / 1e6, color="C0")
ax_tw = ax.twinx()
ax_tw.plot(x_3dec, -sn_3dec / 1e6, ".", markevery=0.1, color="C1")
ax_tw.plot(
    x_hat[x_hat >= 0] * params.L, -sn_hat[x_hat >= 0] * sn_char / 1e6, color="C1"
)
ax.set(
    xlabel="Distance $x$ (m)",
    ylabel="Pressure $p$ (MPa)",
    xlim=(0, 100),
    ylim=(0, None),
)
ax_tw.set(ylabel=r"Tensile stress $-\sigma_{n}$ (MPa)")
ax.text(0.97, 0.95, f"$t = {t:g}$ s", transform=ax.transAxes, ha="right", va="top")
# style legend, shared by both axes
ax_tw.legend(
    handles=[
        Line2D([], [], ls="none", marker=".", color="0.3", label="3DEC"),
        Line2D([], [], color="0.3", label="Semi-analytical"),
    ],
    loc="lower right",
    bbox_to_anchor=(1, 0.08),
    fontsize="x-small",
)
# colour each y-axis (spine, ticks, label) like its curve
ax.spines["right"].set_visible(False)
ax_tw.spines["left"].set_visible(False)
for a, side, c in ((ax, "left", "C0"), (ax_tw, "right", "C1")):
    a.spines[side].set_color(c)
    a.tick_params(axis="y", colors=c, which="both")
    a.yaxis.label.set_color(c)
fig.canvas.draw_idle()
plt.pause(0.01)

# %%

plotting.save_figure(fig, "pressure-stress-front", overleaf=True)


# %%
