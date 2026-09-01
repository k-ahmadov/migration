# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fracinj import paths
from fracinj.analysis import RIGID, SOFT
from fracinj.math_utils import fit_power_law_fixed_exponent

SECONDS_PER_DAY = 86_400


# %%
def load_soultz(filepath):
    df = pd.read_csv(filepath, sep=";")
    fmt = "%d/%m/%Y %H:%M:%S.%f"
    timestamps = pd.to_datetime(
        df["Date (local)"] + " " + df["Time (local)"], format=fmt
    )
    date_mask = timestamps < timestamps.iloc[0] + pd.Timedelta(days=18)
    df, timestamps = df.loc[date_mask], timestamps[date_mask]
    t = (timestamps - timestamps.iloc[0]).dt.total_seconds()
    x, y, z = df["East (m)"], df["North (m)"], df["Depth (m)"]
    x0, y0, z0 = x.iloc[:10].median(), y.iloc[:10].median(), z.iloc[:10].median()
    distance = np.sqrt((x - x0) ** 2 + (y - y0) ** 2 + (z - z0) ** 2)
    return t.values, distance.values


def load_basel(filepath):
    df = pd.read_csv(
        filepath,
        comment="#",
        sep=r"\s+",
        names=[
            "SourceDateTime", "LSrc", "Lat", "Lon", "Dep", "X", "Y", "Z",
            "Mwx", "MwGEL", "MwSED", "MLSED", "ID", "TpID", "GELID", "SEDID",
        ],
    )
    df.replace(
        {"-.--": float("nan"), "-.-": float("nan"), r"\-{3,}": float("nan")},
        inplace=True,
        regex=True,
    )
    df["SourceDateTime"] = pd.to_datetime(df["SourceDateTime"])
    for col in ["Lat", "Lon", "Dep", "X", "Y", "Z", "Mwx", "MwGEL", "MwSED", "MLSED"]:
        df[col] = pd.to_numeric(df[col])
    date_mask = (df["SourceDateTime"] > pd.Timestamp(2006, 12, 2)) & (
        df["SourceDateTime"] < pd.Timestamp(2006, 12, 11)
    )
    df = df.loc[date_mask].reset_index(drop=True)
    timestamps = df["SourceDateTime"]
    t = (timestamps - timestamps.iloc[0]).dt.total_seconds()
    x, y, z = df["X"], df["Y"], df["Z"]
    x0, y0, z0 = x.iloc[:10].median(), y.iloc[:10].median(), z.iloc[:10].median()
    distance = np.sqrt((x - x0) ** 2 + (y - y0) ** 2 + (z - z0) ** 2)
    return t.values, distance.values


def percentile_envelope(t, distance, bin_size, percentile):
    """Pick the ``percentile`` distance (and its time) from each bin of ``bin_size`` events."""
    n_bins = len(distance) // bin_size
    n_trim = n_bins * bin_size
    d_bins = distance[:n_trim].reshape(n_bins, bin_size)
    t_bins = t[:n_trim].reshape(n_bins, bin_size)
    p_vals = np.percentile(d_bins, percentile, axis=1, method="nearest")
    idx = np.argmin(np.abs(d_bins - p_vals[:, None]), axis=1)
    rows = np.arange(n_bins)
    return t_bins[rows, idx], d_bins[rows, idx]


def analytical_fit_and_split(t_pct, d_pct, split_frac):
    """Split into early/late series and fit a fixed-exponent power law to each."""
    idx_start = int(len(t_pct) * split_frac)
    t_early, d_early = t_pct[:idx_start], d_pct[:idx_start]
    t_late, d_late = t_pct[idx_start:], d_pct[idx_start:]
    A_e = fit_power_law_fixed_exponent(t_early, d_early, RIGID.front_exponent)
    A_l = fit_power_law_fixed_exponent(t_late, d_late, SOFT.front_exponent)
    return (
        (t_early, A_e * t_early**RIGID.front_exponent, RIGID.front_exponent, A_e),
        (t_late, A_l * t_late**SOFT.front_exponent, SOFT.front_exponent, A_l),
    )


def _to_days(t_sec):
    return t_sec / SECONDS_PER_DAY


def plot_front(
    ax, t, distance, t_pct, d_pct, early, late, bin_size, percentile, title,
    xlim, ylim, early_ann_idx=10, late_ann_idx=40,
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
        arrowprops=dict(arrowstyle="<-", shrinkA=2, shrinkB=2),
    )
    ax.plot(_to_days(t_l), d_fit_l, "--", color="tab:blue", lw=3)
    ax.annotate(
        "Late-time\n" + rf"$x_f = {A_l:.2f} t^{{{alpha_l:.2f}}}$",
        xy=(_to_days(t_l)[late_ann_idx], d_fit_l[late_ann_idx]),
        xytext=(-80, 20),
        textcoords="offset points",
        ha="left",
        backgroundcolor="1",
        arrowprops=dict(arrowstyle="<-", shrinkA=2, shrinkB=2),
    )
    ax.set(xlabel="Time [day]", title=title, xscale="log", yscale="log", xlim=xlim, ylim=ylim)
    ax.spines[["right", "top"]].set_visible(False)
    leg = ax.legend(loc="lower right", framealpha=1, edgecolor="0.5")
    leg.legend_handles[0].set_markersize(7)


# %% ---- Data ----------------------------------------------------------

t_s, d_s = load_soultz(paths.DATA / "SSFS1993-Catalogue_Bourouis.csv")
t_pct_s, d_pct_s = percentile_envelope(t_s, d_s, bin_size=50, percentile=90)
early_s, late_s = analytical_fit_and_split(t_pct_s, d_pct_s, split_frac=1 / 4)

t_b, d_b = load_basel(paths.DATA / "supp2_compilation_existing_catalogs.dat")
t_pct_b, d_pct_b = percentile_envelope(t_b, d_b, bin_size=50, percentile=75)
early_b, late_b = analytical_fit_and_split(t_pct_b, d_pct_b, split_frac=1 / 4)

# %% ---- Combined figure ---------------------------------------------

fig = plt.figure(
    figsize=(6.4 * 1.25, 4.8 / 1.5), dpi=150, layout="constrained", clear=True, num=1
)
ax1, ax2 = fig.subplots(1, 2)

plot_front(
    ax1, t_s, d_s, t_pct_s, d_pct_s, early_s, late_s,
    bin_size=50, percentile=90, title="Soultz 1993",
    xlim=(0.5, 20), ylim=(50, 1100), early_ann_idx=10, late_ann_idx=40,
)
plot_front(
    ax2, t_b, d_b, t_pct_b, d_pct_b, early_b, late_b,
    bin_size=50, percentile=75, title="Basel 2006",
    xlim=(0.5, 10), ylim=(50, 1000), early_ann_idx=3, late_ann_idx=10,
)

ax1.set_ylabel("Distance [m]")
ax1.annotate("(a)", (0, 1.03), xycoords="axes fraction", fontsize="large")
ax2.annotate("(b)", (0, 1.03), xycoords="axes fraction", fontsize="large")

fig.canvas.draw_idle()
plt.pause(0.01)

# plotting.save_figure(fig, "Fig5", overleaf=True)
