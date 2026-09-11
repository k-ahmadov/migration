import matplotlib.pyplot as plt

from fracinj import paths
from fracinj.analysis import analytical_fit_and_split
from fracinj.detection import percentile_envelope
from fracinj.io import _load_basel, _load_soultz

# %%

SECONDS_PER_DAY = 86_400


def _to_days(t_sec):
    return t_sec / SECONDS_PER_DAY


def plot_front(
    ax,
    t,
    distance,
    t_pct,
    d_pct,
    early,
    late,
    bin_size,
    percentile,
    title,
    xlim,
    ylim,
    early_ann_idx=10,
    late_ann_idx=40,
):
    t_e, d_fit_e, alpha_e, A_e = early
    t_l, d_fit_l, alpha_l, A_l = late

    ax.plot(_to_days(t), distance, ".", color="tab:gray", ms=3, label="Seismic events")
    ax.plot(_to_days(t_pct), d_pct, "ko", label=f"P{percentile} per {bin_size} events")
    ax.plot(_to_days(t_e), d_fit_e, "--", color="tab:blue", lw=3, label="Power-law fit")
    ax.annotate(
        "Early-time\n" + rf"$x_f = {A_e:.2f} t^{{{alpha_e:.2f}}}$",
        xy=(_to_days(t_e)[early_ann_idx], d_fit_e[early_ann_idx]),
        xytext=(-80, 40),
        textcoords="offset points",
        ha="left",
        backgroundcolor="1",
        arrowprops={"arrowstyle": "<-", "shrinkA": 2, "shrinkB": 2},
    )
    ax.plot(_to_days(t_l), d_fit_l, "--", color="tab:blue", lw=3)
    ax.annotate(
        "Late-time\n" + rf"$x_f = {A_l:.2f} t^{{{alpha_l:.2f}}}$",
        xy=(_to_days(t_l)[late_ann_idx], d_fit_l[late_ann_idx]),
        xytext=(-80, 20),
        textcoords="offset points",
        ha="left",
        backgroundcolor="1",
        arrowprops={"arrowstyle": "<-", "shrinkA": 2, "shrinkB": 2},
    )
    ax.set(
        xlabel="Time [day]",
        title=title,
        xscale="log",
        yscale="log",
        xlim=xlim,
        ylim=ylim,
    )
    ax.spines[["right", "top"]].set_visible(False)
    leg = ax.legend(loc="lower right", framealpha=1, edgecolor="0.5")
    leg.legend_handles[0].set_markersize(7)


# %% ---- Data ----------------------------------------------------------

t_s, d_s = _load_soultz(paths.DATA / "SSFS1993-Catalogue_Bourouis.csv")
t_pct_s, d_pct_s = percentile_envelope(t_s, d_s, bin_size=50, percentile=90)
early_s, late_s = analytical_fit_and_split(t_pct_s, d_pct_s, split_frac=1 / 4)

t_b, d_b = _load_basel(paths.DATA / "supp2_compilation_existing_catalogs.dat")
t_pct_b, d_pct_b = percentile_envelope(t_b, d_b, bin_size=50, percentile=75)
early_b, late_b = analytical_fit_and_split(t_pct_b, d_pct_b, split_frac=1 / 4)

# %% ---- Combined figure ---------------------------------------------

fig = plt.figure(
    figsize=(6.4 * 1.25, 4.8 / 1.5), dpi=150, layout="constrained", clear=True, num=1
)
ax1, ax2 = fig.subplots(1, 2)

plot_front(
    ax1,
    t_s,
    d_s,
    t_pct_s,
    d_pct_s,
    early_s,
    late_s,
    bin_size=50,
    percentile=90,
    title="Soultz 1993",
    xlim=(0.5, 20),
    ylim=(50, 1100),
    early_ann_idx=10,
    late_ann_idx=40,
)
plot_front(
    ax2,
    t_b,
    d_b,
    t_pct_b,
    d_pct_b,
    early_b,
    late_b,
    bin_size=50,
    percentile=75,
    title="Basel 2006",
    xlim=(0.5, 10),
    ylim=(50, 1000),
    early_ann_idx=3,
    late_ann_idx=10,
)

ax1.set_ylabel("Distance [m]")
ax1.annotate("(a)", (0, 1.03), xycoords="axes fraction", fontsize="large")
ax2.annotate("(b)", (0, 1.03), xycoords="axes fraction", fontsize="large")

fig.canvas.draw_idle()
plt.pause(0.01)

# plotting.save_figure(fig, "Fig5", overleaf=True)
