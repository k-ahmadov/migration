# %%
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple

from mypackages import file_io, front_analysis, plotting

# %%


def plot_constant_rate(
    ax,
    runs: dict[float, file_io.RunData],
    results: dict[float, front_analysis.AnalyticalFrontResults],
    configs: dict[float, Any],
    log_scale: bool,
):
    for q in configs.keys():
        idx = len(results[q].t_front) //  int(configs[q]["ann_pos_frac"])
        ax.plot(
            results[q].t_front,
            results[q].x_front,
            configs[q]["marker"],
            color="tab:gray",
            markevery=1,
            label="Numerical",
        )
        if configs[q]["fit_type"] == "Analytical":
            ax.plot(
                results[q].t_front, results[q].x_analytical(), "k-", label="Analytical"
            )
        elif configs[q]["fit_type"] == "Empirical":
            ax.plot(results[q].t_front, results[q].x_empirical(), "k--", label="Empirical")
        if log_scale:
            plotting.slope_triangle(
                ax,
                results[q].t_front[idx],
                prefactor=results[q].A_ana
                if configs[q]["fit_type"] == "Analytical"
                else results[q].A_emp,
                slope=results[q].alpha_ana
                if configs[q]["fit_type"] == "Analytical"
                else round(results[q].alpha_emp, 1),
                dx_log=0.5,
            )
            ax.annotate(
                rf"$q_0={runs[q].params.flux:.0e}~\mathsf{{m^2/s}}$",
                xy=(
                    results[q].t_front[idx],
                    results[q].x_front[idx],
                ),
                xytext=configs[q]["xytext"],
                textcoords="offset points",
                ha="left" if configs[q]["xytext"][1] < 0 else "right",
                arrowprops=dict(arrowstyle="<-", shrinkA=4, shrinkB=4),
            )
    ax.set(
        xlabel="Time $t$, [s]",
        ylabel="Front position $x_f$, [m]",
        xscale="log" if log_scale else "linear",
        yscale="log" if log_scale else "linear",
        title = "Log scale" if log_scale else "Linear scale"
    )
    return ax


# %%

result_dir = Path.cwd() / "results" / "3dec" / "runs-wi-1e-05"

run_configs = {
    5e-5: dict(
        analysis_fn=front_analysis.analyze_soft,
        marker="x",
        xytext=(0, 20),
        fit_type="Analytical",
        ann_pos_frac = 20
    ),
    5e-7: dict(
        analysis_fn=front_analysis.analyze,
        marker="1",
        xytext=(-25, -80),
        fit_type="Empirical",
        ann_pos_frac = 20
    ),
    5e-9: dict(
        analysis_fn=front_analysis.analyze_rigid,
        marker=".",
        xytext=(-40, -50),
        fit_type="Analytical",
        ann_pos_frac = 150 
    ),
}

runs = {q: file_io.read_run(result_dir / f"run-q-{q:.0e}.hdf5") for q in run_configs}
results = {}
for q, cfg in run_configs.items():
    run = runs[q]
    kwargs: dict[str, Any] = {"stress_front": True}
    if cfg["analysis_fn"] is front_analysis.analyze_rigid:
        kwargs["slc"] = slice(None, len(run.t) // 3)
    results[q] = cfg["analysis_fn"](run, **kwargs)

# %%

fig = plt.figure(
    figsize=(6.4 * 1.25, 4.8 / 1.5), dpi=150, layout="constrained", clear=True, num=1
)
axes = fig.subplots(1, 2)
ax1, ax2 = axes[0], axes[1]

plot_constant_rate(
    ax=ax1, runs=runs, results=results, configs=run_configs, log_scale=False
)

plot_constant_rate(
    ax=ax2, runs=runs, results=results, configs=run_configs, log_scale=True
)

ax1.annotate(
    r"$x_f=\zeta_f \ (D t)^{1/2}$" "\n(rigid)",
    xy=(results[5e-9].t_front[len(results[5e-9].t_front)//2], results[5e-9].x_analytical()[len(results[5e-9].t_front)//2]),
    xytext=(40, -35),
    textcoords="offset points",
    ha="left",
    arrowprops=dict(arrowstyle="<-", shrinkA=4, shrinkB=4),
)
ax1.annotate(
    r"$x_f=\zeta_f \ ( M q^3 t^4)^{1/5}$" "\n(soft)",
    xy=(results[5e-5].t_front[len(results[5e-5].t_front)//2], results[5e-5].x_analytical()[len(results[5e-5].t_front)//2]),
    xytext=(15, 30),
    textcoords="offset points",
    ha="left",
    arrowprops=dict(arrowstyle="<-", shrinkA=4, shrinkB=4),
)
ax1.annotate(
    r"$x_f \sim t^{0.7}$ (intermediate)",
    xy=(results[5e-7].t_front[len(results[5e-7].t_front)//4], results[5e-7].x_empirical()[len(results[5e-7].t_front)//4]),
    xytext=(40, 0),
    textcoords="offset points",
    ha="left",
    arrowprops=dict(arrowstyle="<-", shrinkA=4, shrinkB=4),
)


# Deduplicate legend entries (both loop iterations add Numerical/Analytical)
handles, labels = ax1.get_legend_handles_labels()
by_label = dict(zip(labels, handles))  # remove duplicate labels
ax1.legend(
    handles=[(handles[0], handles[2], handles[4]), handles[1], handles[3]],
    labels=by_label.keys(),
    loc="upper right",
    handler_map={tuple: HandlerTuple(ndivide=None)},
)

ax1.annotate("(a)", (-0.1, 1.03), xycoords="axes fraction", fontsize="large")
ax2.annotate("(b)", (-0.1, 1.03), xycoords="axes fraction", fontsize="large")

fig.canvas.draw_idle()
plt.pause(0.01)

# plt.savefig(Path.cwd()/ 'figures'/ 'paper' / 'Fig4.eps')
# plt.savefig(Path.cwd()/ 'figures'/ 'front' / 'linear-vs-log.png')
# plt.savefig(Path.cwd() / "overleaf" / "figures_main" / "Fig4.eps")



# %%
# # arrow base and direction: 45 degrees left of north (i.e. northwest), pointing away from base
# x0, y0 = 1.5, 0.3
# length = 1
# angle_deg = 125  # 90 = straight up (north); +45 rotates left (counter-clockwise) to NW
# angle_rad = np.deg2rad(angle_deg)
# dx, dy = length * np.cos(angle_rad), length * np.sin(angle_rad)
#
# ax2.annotate(
#     "",
#     xy=(x0 + dx, y0 + dy),      # arrow tip
#     xytext=(x0, y0),            # arrow base
#     arrowprops=dict(arrowstyle="->", color="k"),
# )
#
# # label near the arrow, offset slightly perpendicular to it so it doesn't overlap
# ax2.text(
#     x0 + dx * 0.6 + 0.3, y0 + dy * 0.6 + 0.02,
#     r"$q_0\uparrow$",
#     fontsize=14,
#     ha="left", va="top",
# )

