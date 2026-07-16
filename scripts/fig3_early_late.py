# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mypackages import file_io, front_analysis, p_inj_analysis, physics, plotting
from mysolvers.exact_solutions import solve_linear_diffusion_const_flux

# %% --- Data ---

result_dir = Path.cwd() / "results" / "3dec" / "runs-wi-1e-05"
run = file_io.read_run(result_dir / "run-q-5e-05.hdf5")
D = physics.diffusivity(run.params)
t_c = physics.critical_time(run.params)
idx_tc = 30 * (np.searchsorted(run.t, t_c) + 1)


# %%
# front
result = front_analysis.analyze(run, stress_front=True)
result_early = front_analysis.analyze_early_time(
    run, stress_front=True, slc=slice(None, idx_tc)
)
A_late, alpha_late = physics.fit_front_power_law(
    result.t_front[idx_tc:], result.x_front[idx_tc:]
)
result_late = front_analysis.analyze_late_time(
    run, stress_front=True, slc=slice(idx_tc, None)
)


# %%

fig = plt.figure(
    figsize=(6.4, 4.8 / 2), dpi=150, layout="constrained", clear=True, num=1
)
ax1, ax2 = fig.subplots(1, 2)

ax1.plot(result.t_front, result.x_front, ".", color="tab:gray", label="Numerical")
ax1.plot(
    result.t_front,
    result_early.A_ana * result.t_front**result_early.alpha_ana,
    "k-",
    label="Analytical",
)
ax1.plot(
    result.t_front, result_late.A_ana * result.t_front**result_late.alpha_ana, "k-"
)

ax2.plot(result.t_front, result.x_front, ".", color="tab:gray", label="Numerical")
ax2.plot(
    result.t_front, result_early.A_ana * result.t_front**result_early.alpha_ana, "k-"
)
ax2.annotate(
    r"$x_f=\zeta_f \ (D t)^{1/2}$" "\n(early-time)",
    xy=(result_early.t_front[3], result_early.x_analytical()[3]),
    xytext=(10, -50),
    textcoords="offset points",
    ha="left",
    arrowprops=dict(arrowstyle="<-", shrinkA=4, shrinkB=4),
)
plotting.slope_triangle(
    ax2,
    result_early.t_front[0],
    prefactor=result_early.A_ana,
    slope=result_early.alpha_ana,
    dx_log=0.5
)

ax2.plot(
    result.t_front, result_late.A_ana * result.t_front**result_late.alpha_ana, "k-"
)
ax2.annotate(
    r"$x_f=\zeta_f \ ( M q^3 t^4)^{1/5}$" "\n(late-time)",
    xy=(result_late.t_front[400], result_late.x_analytical()[400]),
    xytext=(-60, -10),
    textcoords="offset points",
    ha="right",
    arrowprops=dict(arrowstyle="<-", shrinkA=4, shrinkB=4),
)
plotting.slope_triangle(
    ax2,
    result_late.t_front[0],
    prefactor=result_late.A_ana,
    slope=result_late.alpha_ana,
)

ax1.set(
    xlabel="Time, $t$ [s]",
    ylabel="Front position $x_f$ [m]",
)
ax1.legend()
ax2.set(
    xscale="log",
    yscale="log",
    xlabel="Time, $t$ [s]",
    ylabel="Front position $x_f$ [m]",
)
fig.canvas.draw_idle()
plt.pause(0.01)

# plt.savefig(Path.cwd()/ 'figures'/ 'paper' / 'Fig3.eps')
# plt.savefig(Path.cwd()/ 'figures'/ 'early-late' / 'q-5e-07-linear-vs-log.png')
# plt.savefig(Path.cwd() / "overleaf" / "figures_main" / "Fig3.eps")
