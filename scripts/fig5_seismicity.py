import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.axes import Axes

from fracinj import analysis, io, plotting

SECONDS_PER_DAY = 86_400
COLUMN_WIDTH = 3.25  # in



# %%
def to_days(t_sec):
    return t_sec / SECONDS_PER_DAY


def plot_slope_triangle(
    ax: Axes, x: np.ndarray, y: np.ndarray, x0: float, nudge: float = 1
):
    exponent, log_prefactor = np.polyfit(np.log(x), np.log(y), deg=1)
    plotting.slope_triangle(
        ax=ax,
        x0=x0,
        slope=round(exponent, 2),
        prefactor=np.exp(log_prefactor + nudge),
    )


def annotate_regime_slopes(ax: Axes, case: analysis.FieldCase, anchors, nudges):
    """Draw an early- and a late-time slope triangle on the fitted curve."""
    idx_tc = np.searchsorted(case.t_fit, case.t_c)
    halves = [slice(None, idx_tc), slice(idx_tc, None)]
    for slc, i, nudge in zip(halves, anchors, nudges):
        plot_slope_triangle(
            ax,
            to_days(case.t_fit[slc]),
            case.d_fit[slc],
            x0=to_days(float(case.t_fit[i])),
            nudge=nudge,
        )


def annotate_prefactors(ax: Axes, field_case: analysis.FieldCase):
    ax.text(
        0.05,
        0.8,
        rf"$A_{{\mathrm{{early}}}} = {field_case.A_early:.2g} \pm {field_case.perr[0]:.1g},$"
        "\n"
        rf"$A_{{\mathrm{{late}}}} = {field_case.A_late:.2g} \pm {field_case.perr[1]:.1g}$",
        transform=ax.transAxes,
    )


def annotate_tc(ax: Axes, t_c_days: float, y: float) -> None:
    """Same style as annotate_tc in the numerical figures."""
    ax.axvline(t_c_days, color="tab:gray", ls="--")
    ax.annotate(
        rf"$t_{{*}} = {t_c_days:g}$ days",
        (t_c_days, y),
        color="tab:gray",
        ha="center",
        backgroundcolor="1",
    )


def set_plain_log_ticks(ax: Axes, xticks, yticks):
    """Plain numbers on log axes that span only about one decade."""
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_formatter(mticker.ScalarFormatter())
        axis.set_minor_formatter(mticker.NullFormatter())


def plot_field_case(ax: Axes, field_case: analysis.FieldCase, **set_kwargs):
    ax.scatter(
        to_days(field_case.t),
        field_case.d,
        s=3,
        color="0.7",
        rasterized=True,
        label="Events",
    )
    ax.scatter(
        to_days(field_case.t_env),
        field_case.d_env,
        color="k",
        label="Envelope",
        s=15,
    )
    ax.plot(
        to_days(field_case.t_fit),
        field_case.d_fit,
        color="tab:blue",
        lw=1.5,
        label="Crossover",
    )
    ax.axvline(
        to_days(field_case.t_c),
        color="k",
        ls=":",
        # label=rf"$t_*=\mathrm{{{to_days(field_case.t_c):.2g}\ days}}$",
    )
    ax.set(
        xlabel="Time (days)",
        ylabel="Distance (m)",
        title=field_case.title,
        **set_kwargs,
    )
    ax.spines[["right", "top"]].set_visible(False)
    ax.tick_params(top=False, right=False, which="both")


delta_fit = 0.05

# reuse the crossover sharpness found from the numerical simulations
crossover_model = analysis.CrossoverModel(delta=delta_fit)

soultz = analysis.fit_field_crossover(
    *io.load("soultz"),
    title="Soultz 1993",
    percentile=90,
    t_c=3 * SECONDS_PER_DAY,
    model=crossover_model,
)
basel = analysis.fit_field_crossover(
    *io.load("basel"),
    title="Basel 2006",
    percentile=75,
    t_c=2 * SECONDS_PER_DAY,
    model=crossover_model,
)


# %%
fig = plt.figure(
    figsize=(2 * COLUMN_WIDTH, 1.2 * 0.675 * COLUMN_WIDTH),
    dpi=150,
    layout="constrained",
    clear=True,
    num=1,
)
ax1, ax2 = fig.subplots(1, 2)

plot_field_case(
    ax1, soultz, xscale="log", yscale="log", xlim=(0.5, 20), ylim=(70, 1200)
)
plot_field_case(ax2, basel, xscale="log", yscale="log", xlim=(0.5, 6.5), ylim=(40, 700))

annotate_regime_slopes(ax1, soultz, anchors=[10, -65], nudges=[1.0, -0.6])
annotate_regime_slopes(ax2, basel, anchors=[70, -50], nudges=[0.8, -0.5])

# annotate_prefactors(ax1, soultz)
# annotate_prefactors(ax2, basel)

set_plain_log_ticks(ax1, xticks=[0.5, 1, 2, 5, 10, 20], yticks=[100, 200, 500, 1000])
set_plain_log_ticks(ax2, xticks=[0.5, 1, 2, 5], yticks=[50, 100, 200, 500])


annotate_tc(ax1, to_days(soultz.t_c), y=800)
annotate_tc(ax2, to_days(basel.t_c), y=450)

ax1.annotate("(a)", (0, 1.05), xycoords="axes fraction")
ax2.annotate("(b)", (0, 1.05), xycoords="axes fraction")

# fig.legend(
#     loc="lower right",
#     framealpha=1,
#     fontsize="smaller",
#     frameon=True,
#     fancybox=True,
#     ncols=3,
# )

handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, loc="outside lower center", ncols=3, frameon=False)

fig.canvas.draw_idle()
plt.pause(0.01)
# %%

plotting.save_figure(fig, "seismicity-migration", ext="pdf")
