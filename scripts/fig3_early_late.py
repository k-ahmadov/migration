# %%
import matplotlib.pyplot as plt
import numpy as np

from fracinj import analysis, io, paths, physics, plotting

# %% --- Data -------------------------------------------------------------

run = io.read_hdf5(paths.results_dir("3dec", "runs-wi-1e-05") / "run-q-5e-05.hdf5")
t_c = physics.critical_time(run.params)
idx_tc = 30 * (int(np.searchsorted(run.t, t_c)) + 1)

numerical = analysis.analyze_front(run, stress_front=True)
early = analysis.analyze_front(
    run, regime=analysis.RIGID, stress_front=True, slc=slice(None, idx_tc)
)
late = analysis.analyze_front(
    run, regime=analysis.SOFT, stress_front=True, slc=slice(idx_tc, None)
)

# %% --- Figure ---------------------------------------------------------

fig = plt.figure(figsize=(6.4, 4.8 / 2), dpi=150, layout="constrained", clear=True, num=1)
ax1, ax2 = fig.subplots(1, 2)

for ax in (ax1, ax2):
    ax.plot(numerical.t, numerical.y, ".", color="tab:gray", label="Numerical")
    ax.plot(numerical.t, early.analytical(numerical.t), "k-", label="Analytical")
    ax.plot(numerical.t, late.analytical(numerical.t), "k-")

ax2.annotate(
    r"$x_f=\zeta_f \ (D t)^{1/2}$" "\n(early-time)",
    xy=(early.t[3], early.analytical()[3]),
    xytext=(10, -50),
    textcoords="offset points",
    ha="left",
    arrowprops=dict(arrowstyle="<-", shrinkA=4, shrinkB=4),
)
plotting.slope_triangle(
    ax2, early.t[0], prefactor=early.A_ana, slope=early.alpha_ana, dx_log=0.5
)
ax2.annotate(
    r"$x_f=\zeta_f \ ( M q^3 t^4)^{1/5}$" "\n(late-time)",
    xy=(late.t[400], late.analytical()[400]),
    xytext=(-60, -10),
    textcoords="offset points",
    ha="right",
    arrowprops=dict(arrowstyle="<-", shrinkA=4, shrinkB=4),
)
plotting.slope_triangle(ax2, late.t[0], prefactor=late.A_ana, slope=late.alpha_ana)

ax1.set(xlabel="Time, $t$ [s]", ylabel="Front position $x_f$ [m]")
ax1.legend()
ax2.set(
    xscale="log",
    yscale="log",
    xlabel="Time, $t$ [s]",
    ylabel="Front position $x_f$ [m]",
)
fig.canvas.draw_idle()
plt.pause(0.01)

# plotting.save_figure(fig, "Fig3", overleaf=True)
