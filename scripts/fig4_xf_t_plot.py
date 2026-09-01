# %%
from dataclasses import dataclass

import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple

from fracinj import analysis, io, paths, plotting
from fracinj.analysis import RIGID, SOFT, Fit, Regime

# %%


@dataclass
class Case:
    regime: Regime | None  # None -> empirical fit only
    marker: str
    xytext: tuple[int, int]
    ann_pos_frac: int
    early_only: bool = False

    @property
    def kind(self) -> str:
        return "Empirical" if self.regime is None else "Analytical"


CASES: dict[float, Case] = {
    5e-5: Case(SOFT, marker="x", xytext=(0, 20), ann_pos_frac=20),
    5e-7: Case(None, marker="1", xytext=(-25, -80), ann_pos_frac=20),
    5e-9: Case(RIGID, marker=".", xytext=(-40, -50), ann_pos_frac=150, early_only=True),
}


def model_curve(fit: Fit, case: Case):
    return fit.empirical() if case.regime is None else fit.analytical()


def model_prefactor_slope(fit: Fit, case: Case) -> tuple[float, float]:
    if case.regime is None:
        return fit.A_emp, round(fit.alpha_emp, 1)
    return fit.A_ana, fit.alpha_ana


def plot_constant_rate(ax, runs, results, log_scale: bool):
    for q, case in CASES.items():
        fit = results[q]
        idx = len(fit.t) // case.ann_pos_frac
        ax.plot(fit.t, fit.y, case.marker, color="tab:gray", label="Numerical")
        style = "k--" if case.regime is None else "k-"
        ax.plot(fit.t, model_curve(fit, case), style, label=case.kind)

        if log_scale:
            prefactor, slope = model_prefactor_slope(fit, case)
            plotting.slope_triangle(ax, fit.t[idx], prefactor=prefactor, slope=slope, dx_log=0.5)
            ax.annotate(
                rf"$q_0={runs[q].params.q_0:.0e}~\mathsf{{m^2/s}}$",
                xy=(fit.t[idx], fit.y[idx]),
                xytext=case.xytext,
                textcoords="offset points",
                ha="left" if case.xytext[1] < 0 else "right",
                arrowprops=dict(arrowstyle="<-", shrinkA=4, shrinkB=4),
            )
    scale = "log" if log_scale else "linear"
    ax.set(
        xlabel="Time $t$, [s]",
        ylabel="Front position $x_f$, [m]",
        xscale=scale,
        yscale=scale,
        title="Log scale" if log_scale else "Linear scale",
    )


# %%

result_dir = paths.results_dir("3dec", "runs-wi-1e-05")
runs = {q: io.read_hdf5(result_dir / f"run-q-{q:.0e}.hdf5") for q in CASES}

results: dict[float, Fit] = {}
for q, case in CASES.items():
    run = runs[q]
    slc = slice(None, len(run.t) // 3) if case.early_only else None
    results[q] = analysis.analyze_front(run, regime=case.regime, stress_front=True, slc=slc)

# %%

fig = plt.figure(
    figsize=(6.4 * 1.25, 4.8 / 1.5), dpi=150, layout="constrained", clear=True, num=1
)
ax1, ax2 = fig.subplots(1, 2)

plot_constant_rate(ax1, runs, results, log_scale=False)
plot_constant_rate(ax2, runs, results, log_scale=True)


def _mid(fit: Fit, frac: int, curve):
    i = len(fit.t) // frac
    return (fit.t[i], curve(fit)[i])


ax1.annotate(
    r"$x_f=\zeta_f \ (D t)^{1/2}$" "\n(rigid)",
    xy=_mid(results[5e-9], 2, lambda f: f.analytical()),
    xytext=(40, -35),
    textcoords="offset points",
    ha="left",
    arrowprops=dict(arrowstyle="<-", shrinkA=4, shrinkB=4),
)
ax1.annotate(
    r"$x_f=\zeta_f \ ( M q^3 t^4)^{1/5}$" "\n(soft)",
    xy=_mid(results[5e-5], 2, lambda f: f.analytical()),
    xytext=(15, 30),
    textcoords="offset points",
    ha="left",
    arrowprops=dict(arrowstyle="<-", shrinkA=4, shrinkB=4),
)
ax1.annotate(
    r"$x_f \sim t^{0.7}$ (intermediate)",
    xy=_mid(results[5e-7], 4, lambda f: f.empirical()),
    xytext=(40, 0),
    textcoords="offset points",
    ha="left",
    arrowprops=dict(arrowstyle="<-", shrinkA=4, shrinkB=4),
)

handles, labels = ax1.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
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

# plotting.save_figure(fig, "Fig4", overleaf=True)
