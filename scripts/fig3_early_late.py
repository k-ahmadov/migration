# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mypackages import file_io, front_analysis, p_inj_analysis, physics, plotting
from mysolvers.exact_solutions import solve_linear_diffusion_const_flux

# %% --- Data ---

result_dir = Path.cwd() / "results" / "3dec" / "runs-wi-1e-05"
run = file_io.read_run(result_dir / "run-q-5e-07.hdf5")
D = physics.diffusivity(run.params)
t_c = physics.characteristic_time(run.params)
idx_tc = 30 * (np.searchsorted(run.t, t_c) + 1)


# %%
# front
result = front_analysis.analyze(run, stress_front=True)
result_early = front_analysis.analyze_early_time(
    run, stress_front=True, slc=slice(None, idx_tc)
)
A_late, α_late = physics.fit_front_power_law(
    result.t_front[idx_tc:], result.x_front[idx_tc:]
)
result_late = front_analysis.analyze_late_time(
    run, stress_front=True, slc=slice(idx_tc, None)
)

# %% pressure at injection point
x_sc, p = file_io.sort_fields(run.x_sc, run.p)
p_inj = p[:, 0]
ζ_ana = solve_linear_diffusion_const_flux(np.linspace(0, 10, 200))[0]
p_inj_ana = ζ_ana * run.params.flux * np.sqrt(run.t / D) * run.params.k_n
A_p_late, α_p_late = physics.fit_front_power_law(run.t[idx_tc:], p_inj[idx_tc:] / 1e6)
result_p_inj_rigid = p_inj_analysis.analyze_rigid(run)
result_p_inj_soft = p_inj_analysis.analyze_soft(run)

# %% --- Plot ---

# plt.ion()
fig, (ax1, ax2) = plt.subplots(
    1,
    2,
    figsize=(6.4 / 1.5 * 2, 4.8 / 1.5),
    dpi=150,
    layout="constrained",
    clear=True,
    num=1,
)

# %% --- front position ---
ax1.cla()
ax2.cla()
ax1.plot(result.t_front, result.x_front, ".", color="tab:gray", label="Numerical")
ax1.plot(
    result.t_front,
    result_early.A_ana * result.t_front**result_early.α_ana,
    "k-",
    label="Analytical",
)
ax1.annotate(
    r"$x_f=\zeta_f \ (D t)^{1/2}$" "\n(early-time)",
    xy=(result_early.t_front[5], result_early.x_analytical()[5]),
    xytext=(40, -20),
    textcoords="offset points",
    ha="left",
    arrowprops=dict(arrowstyle="<-", shrinkA=4, shrinkB=4),
)
ax1.plot(result.t_front, result_late.A_ana * result.t_front**result_late.α_ana, "k-")
ax1.annotate(
    r"$x_f=\zeta_f \ ( M q^3 t^4)^{1/5}$" "\n(late-time)",
    xy=(result_late.t_front[250], result_late.x_analytical()[250]),
    xytext=(-30, 0),
    textcoords="offset points",
    ha="right",
    arrowprops=dict(arrowstyle="<-", shrinkA=4, shrinkB=4),
)

plotting.slope_triangle(
    ax1, result_early.t_front[1], prefactor=result_early.A_ana, slope=result_early.α_ana
)
plotting.slope_triangle(
    ax1, result_late.t_front[2], prefactor=result_late.A_ana, slope=result_late.α_ana
)

ax1.set(
    xscale="log",
    yscale="log",
    xlabel="Time, $t$ [s]",
    ylabel="Front position $x_f$ [m]",
    ylim=(1, 110),
)
ax1.legend()

# --- pressure at injection point ---
ax2.plot(run.t, p[:, 0], ".", color="tab:gray", label="Numerical")

ax2.plot(run.t, result_p_inj_rigid.p_inj_analytical(run.t), "k-")
ax2.annotate(
    r"$p_{{\mathrm{{inj}}}} =\theta_{{\mathrm{{inj}}}}\ k_n \, q\, D^{-1/2}\, t^{1/2}$"
    "\n(early-time)",
    xy=(run.t[5], p_inj_ana[5]),
    xytext=(25, 30),
    textcoords="offset points",
    ha="right",
    arrowprops=dict(arrowstyle="<-", shrinkA=4, shrinkB=4),
)

ax2.plot(run.t, result_p_inj_soft.p_inj_analytical(run.t), "k", label="analytical")
ax2.annotate(
    r"$p_{\mathrm{inj}}=\theta_{\mathrm{inj}}\ k_n \, (q^2 t/ M)^{1/5} $"
    "\n (late-time)",
    xy=(run.t[100], (result_p_inj_soft.p_inj_analytical(run.t))[100]),
    xytext=(100, 25),
    textcoords="offset points",
    ha="right",
    arrowprops=dict(arrowstyle="<-", shrinkA=4, shrinkB=4),
)

ax2.set(
    xscale="log",
    yscale="log",
    xlabel="Time $t$, [s]",
    ylabel=r"$p_{\mathrm{inj}}$ [Pa]",
)
ax2.legend()
ax1.set_title(rf"Applied injection rate $q={run.params.flux:.0e}~\mathsf{{m^2/s}}$")

# plt.savefig(Path.cwd()/ 'figures'/ 'paper' / 'Fig4.eps')
# plt.savefig(Path.cwd()/ 'figures'/ 'early-late' / 'q-5e-07.png')
# plt.savefig(Path.cwd() / "overleaf" / "figures_main" / "Fig4.eps")

plt.pause(0.01)

# %%
# ax2.plot(run.t, A_p_late * 1e6 * run.t**α_p_late, "k--", label="Fit")
# ax2.annotate(
#     rf"$p_0=A t^{{{α_p_late:.2f}}}$",
#     xy=(run.t[200], (A_p_late * 1e6 * run.t**α_p_late)[200]),
#     xytext=(10, -40),
#     textcoords="offset points",
#     ha="left",
#     arrowprops=dict(arrowstyle="<-", shrinkA=4, shrinkB=4),
# )
