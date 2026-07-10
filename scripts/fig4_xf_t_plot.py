# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.legend_handler import HandlerTuple

from mypackages import file_io, front_analysis, front_detection, physics, plotting
from mypackages.types import HydraulicDiffusivity


# %%
def plot_constant_pressure(
    ax,
    results: list[front_analysis.FrontResultsWithAnalytical],
    runs: list[file_io.RunData],
    xytexts: list[tuple[int, int]],
    marker_styles: list[str],
):
    for result, run, xytext, marker_style in zip(results, runs, xytexts, marker_styles):
        D = (result.A_ana / result.ζ_front) ** (1 / result.α_ana)
        idx = len(result.t_front) // 8

        ax.plot(
            result.t_front,
            result.x_front,
            marker_style,
            color="tab:gray",
            label="Numerical",
        )
        ax.plot(
            result.t_front, result.x_analytical(), "-", color="k", label="Analytical"
        )
        plotting.slope_triangle(
            ax,
            x0=result.t_front[idx],
            prefactor=result.A_ana,
            slope=result.α_ana,
            dx_log=0.3,
        )
        ax.annotate(
            text=(
                rf"$x_f = \zeta_f \, \sqrt{{D t}}$"
                "\n"
                rf"$\zeta_f={result.ζ_front:.1f}$, $D={D:.1f}~\mathsf{{m^2/s}}$"
                "\n"
                rf"$k_n={run.params.k_n / 1e9}~\mathsf{{GPa/m}}$"
            ),
            xy=(result.t_front[idx // 2], result.x_front[idx // 2]),
            xytext=xytext,
            textcoords="offset points",
            ha="right" if xytext[1] > 0 else "left",
            arrowprops=dict(arrowstyle="<-", shrinkA=4, shrinkB=4),
        )
    # Deduplicate legend entries (both loop iterations add Numerical/Analytical)
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))  # remove duplicate labels
    # ax.legend(handles=by_label.values(), labels=by_label.keys(), loc="lower right")
    ax.legend(
        handles=[(handles[0], handles[2]), handles[1]],
        labels=by_label.keys(),
        loc="lower right",
        handler_map={tuple: HandlerTuple(ndivide=None)},
    )
    ax.set(
        xlabel="Time $t$, [s]",
        ylabel="Front position $x_f$, [m]",
        xscale="log",
        yscale="log",
        title="Constant pressure injection",
    )
    return ax


def recalibrate_zeta(
    D: HydraulicDiffusivity, result: front_analysis.FrontResultsWithAnalytical
):
    # recalibrate ζ_front for rigid case to only use the first half of data
    idx = len(result.t_front) // 2
    zeta = float(
        np.mean(result.x_front[:idx] / (D * result.t_front[:idx]) ** result.α_ana)
    )
    return zeta


def plot_constant_rate(
    ax,
    runs: list[file_io.RunData],
    results: list[front_analysis.FrontResultsWithAnalytical],
    marker_styles: list[str],
    xytexts: list[tuple[int, int]],
    fit_types: list[str],
):
    # recalibrate ζ for rigid case to only use the first half of data
    D = physics.diffusivity(runs[0].params)
    ζ = recalibrate_zeta(D, results[0])
    results[0].A_ana = ζ * D ** results[0].α_ana

    for result, run, marker_style, xytext, fit_type in zip(
        results, runs, marker_styles, xytexts, fit_types
    ):
        idx = len(result.t_front) // 10
        ax.plot(
            result.t_front,
            result.x_front,
            marker_style,
            color="tab:gray",
            markevery=2,
            label="Numerical",
        )
        if fit_type == "Analytical":
            ax.plot(result.t_front, result.x_analytical(), "k-", label="Analytical")
        elif fit_type == "Empirical":
            ax.plot(result.t_front, result.x_empirical(), "k--")
        else:
            raise ValueError("incorrect fit type")
        plotting.slope_triangle(
            ax,
            result.t_front[idx],
            prefactor=result.A_ana if fit_type == "Analytical" else result.A_emp,
            slope=result.α_ana if fit_type == "Analytical" else round(result.α_emp, 1),
            dx_log=0.5,
        )
        ax.annotate(
            rf"$q_0={run.params.flux:.0e}~\mathsf{{m^2/s}}$",
            xy=(
                result.t_front[idx],
                result.x_front[idx],
            ),
            xytext=xytext,
            textcoords="offset points",
            ha="left" if xytext[1] < 0 else "right",
            arrowprops=dict(arrowstyle="<-", shrinkA=4, shrinkB=4),
        )

    ax.set(
        xlabel="Time $t$, [s]",
        ylabel="Front position $x_f$, [m]",
        xscale="log",
        yscale="log",
        title="Constant rate injection",
        # ylim=(2, 110)
    )
    # Deduplicate legend entries (both loop iterations add Numerical/Analytical)
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))  # remove duplicate labels
    ax.legend(
        handles=[(handles[0], handles[2], handles[4]), handles[1]],
        labels=by_label.keys(),
        loc="lower right",
        handler_map={tuple: HandlerTuple(ndivide=None)},
    )
    return ax


# --------------------------------------------------------------
# %% for poster
# --------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(
    1,
    2,
    figsize=(6.4 * 1.25, 4.8 / 1.5),
    dpi=200,
    constrained_layout=True,
    num=1,
    clear=True,
)

# TODO: add (a) and (b) annotations to the figure
# %%
# ────────────────────────────────────────────────────────────────
# ── Left panel ──────────────────────────────────────────────────
# ────────────────────────────────────────────────────────────────
ax1.cla()
ax2.cla()
result_dir_cp = Path.cwd() / "results" / "3dec" / "constant-pressure"
filenames_cp = ["rigid.pkl", "soft.pkl"]
runs_cp = [file_io.read_pickle(result_dir_cp / f) for f in filenames_cp]
results_cp = [
    front_analysis.analyze_rigid(runs_cp[0], stress_front=True),
    front_analysis.analyze_rigid(runs_cp[1], stress_front=True),
]
xytexts_cp = [(15, 25), (-35, -65)]
marker_styles = [".", "x"]

plot_constant_pressure(
    ax=ax1,
    results=results_cp,
    runs=runs_cp,
    xytexts=xytexts_cp,
    marker_styles=marker_styles,
)


# ────────────────────────────────────────────────────────────────
# ── Right panel ─────────────────────────────────────────────────
# ────────────────────────────────────────────────────────────────

result_dir_runs = Path.cwd() / "results" / "3dec" / "runs-wi-1e-05"
filenames_runs = [
    "run-q-5e-09.hdf5",
    "run-q-5e-05.hdf5",
    "run-q-5e-07.hdf5",
]
runs_q = [file_io.read_run(result_dir_runs / f) for f in filenames_runs]
results_q = [
    front_analysis.analyze_rigid(runs_q[0], stress_front=True),
    front_analysis.analyze_soft(runs_q[1], stress_front=True),
    front_analysis.analyze(runs_q[2], stress_front=True),
]
marker_styles = [".", "x", "1"]
xytexts_q = [
    (-45, -50),
    (0, 20),
    (-38, -85),
]
fit_types = ["Analytical", "Analytical", "Empirical"]

plot_constant_rate(
    ax=ax2,
    runs=runs_q,
    results=results_q,
    marker_styles=marker_styles,
    xytexts=xytexts_q,
    fit_types=fit_types,
)

ax1.annotate("(a)", (-0.1, 1.03), xycoords="axes fraction", fontsize="large")
ax2.annotate("(b)", (-0.1, 1.03), xycoords="axes fraction", fontsize="large")

# fig.savefig(Path.cwd() / "overleaf" / "figures_main" / "Fig3.eps")

plt.pause(0.01)

# %%
