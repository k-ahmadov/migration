from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mypackages import physics


def load_soultz(filepath, bin_size):
    # load data file
    df = pd.read_csv(filepath, sep=";")
    # formatting string for date and time format
    fmt = "%d/%m/%Y %H:%M:%S.%f"
    # import time
    timestamps = pd.to_datetime(
        df["Date (local)"] + " " + df["Time (local)"], format=fmt
    )
    # convert time to hours
    t_days = (timestamps - timestamps.iloc[0]).dt.total_seconds() / 86400
    # import coordinates of events
    x, y, z = df["East (m)"], df["North (m)"], df["Depth (m)"]
    # median of first 10 events
    x0, y0, z0 = x.iloc[:10].median(), y.iloc[:10].median(), z.iloc[:10].median()
    # euclidian distance
    distance = np.sqrt((x - x0) ** 2 + (y - y0) ** 2 + (z - z0) ** 2)
    return t_days.values, distance.values


def load_basel(filepath):
    # load data file
    df = pd.read_csv(
        filepath,
        comment="#",
        sep=r"\s+",
        names=[
            "SourceDateTime",
            "LSrc",
            "Lat",
            "Lon",
            "Dep",
            "X",
            "Y",
            "Z",
            "Mwx",
            "MwGEL",
            "MwSED",
            "MLSED",
            "ID",
            "TpID",
            "GELID",
            "SEDID",
        ],
    )
    # replace empty data with Nan
    df.replace(
        {"-.--": float("nan"), "-.-": float("nan"), r"\-{3,}": float("nan")},
        inplace=True,
        regex=True,
    )
    # convert to date time format
    df["SourceDateTime"] = pd.to_datetime(df["SourceDateTime"])
    # convert to numeric data
    for col in ["Lat", "Lon", "Dep", "X", "Y", "Z", "Mwx", "MwGEL", "MwSED", "MLSED"]:
        df[col] = pd.to_numeric(df[col])
    # cut data to injection period
    date_mask = (df["SourceDateTime"] > pd.Timestamp(2006, 12, 2)) & (
        df["SourceDateTime"] < pd.Timestamp(2006, 12, 11)
    )
    df = df.loc[date_mask].reset_index(drop=True)
    # import time data
    timestamps = df["SourceDateTime"]
    # convert time from sec to hours
    t_days = (timestamps - timestamps.iloc[0]).dt.total_seconds() / 86400
    # import coordinates of events
    x, y, z = df["X"], df["Y"], df["Z"]
    # median of first 10 events
    x0, y0, z0 = x.iloc[:10].median(), y.iloc[:10].median(), z.iloc[:10].median()
    # euclidian distance
    distance = np.sqrt((x - x0) ** 2 + (y - y0) ** 2 + (z - z0) ** 2)
    return t_days.values, distance.values


def percentile_envelope(t_days, distance, bin_size, percentile):
    "Chooses the distance based on the given percentile per bin"
    # number of bins based on bin size
    n_bins = len(distance) // bin_size
    n_trim = n_bins * bin_size
    # trim distance and time data per bin
    d_bins = distance[:n_trim].reshape(n_bins, bin_size)
    t_bins = t_days[:n_trim].reshape(n_bins, bin_size)
    # choose distance value from each bin according to the given percentile
    p_vals = np.percentile(d_bins, percentile, axis=1, method="nearest")
    # find its indices
    idx = np.argmin(np.abs(d_bins - p_vals[:, None]), axis=1)
    # import the corresponding time and distance values
    t_pct = t_bins[np.arange(n_bins), idx]
    d_pct = d_bins[np.arange(n_bins), idx]
    return t_pct, d_pct


def fit_and_split(t_pct, d_pct, split_frac, t_max=None):
    "Function to extract the early- and late-time series and fit a power law function"
    # filter data according to the give maximum time value
    if t_max is not None:
        keep = t_pct < t_max
        t_pct, d_pct = t_pct[keep], d_pct[keep]
    # index to separate early and late data
    idx_start = int(len(t_pct) * split_frac)
    # import early and late
    t_early, d_early = t_pct[:idx_start], d_pct[:idx_start]
    t_late, d_late = t_pct[idx_start:], d_pct[idx_start:]
    # find best fit power law prefactor and exponent to the early and late time treands
    A_e, a_e = physics.fit_front_power_law(t_early, d_early)
    A_l, a_l = physics.fit_front_power_law(t_late, d_late)
    return (t_early, A_e * t_early**a_e, a_e), (t_late, A_l * t_late**a_l, a_l)


def plot_front(
    ax,
    t_days,
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
    mask=None,
    early_ann_idx=10,
    late_ann_idx=40,
):
    if mask is None:
        mask = np.ones_like(t_days, dtype=bool)
    t_e, d_fit_e, a_e = early
    t_l, d_fit_l, a_l = late

    ax.plot(
        t_days[mask],
        distance[mask],
        ".",
        color="tab:gray",
        ms=3,
        label="Seismic events",
    )
    ax.plot(t_pct, d_pct, "ko", label=f"P{percentile} per {bin_size} events")
    ax.plot(t_e, d_fit_e, "--", color="tab:blue", lw=3, label="Power-law fit")
    ax.annotate(
        "Early-time\n" + rf"$x_f \propto t^{{{a_e:.2f}}}$",
        xy=(t_e[early_ann_idx], d_fit_e[early_ann_idx]),
        xytext=(-20, 30),
        textcoords="offset points",
        ha="right",
        backgroundcolor="1",
        arrowprops=dict(arrowstyle="<-", shrinkA=2, shrinkB=2),
    )
    ax.plot(t_l, d_fit_l, "--", color="tab:blue", lw=3)
    ax.annotate(
        "Late-time\n" + rf"$x_f \propto t^{{{a_l:.2f}}}$",
        xy=(t_l[late_ann_idx], d_fit_l[late_ann_idx]),
        xytext=(-40, 20),
        textcoords="offset points",
        ha="right",
        backgroundcolor="1",
        arrowprops=dict(arrowstyle="<-", shrinkA=2, shrinkB=2),
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


# %% ---- Soultz 1993 ----
t_s, d_s = load_soultz(
    "/home/kahmadov/phd/migration/data/SSFS1993-Catalogue_Bourouis.csv",
    bin_size=50,
    # percentile=90,
)
t_pct_s, d_pct_s = percentile_envelope(t_s, d_s, bin_size=50, percentile=90)
early_s, late_s = fit_and_split(t_pct_s, d_pct_s, split_frac=1 / 5, t_max=18)
mask_s = (t_s <= 18) & (d_s <= 1050)

# %% ---- Basel 2006 ----
t_b, d_b = load_basel(
    "/home/kahmadov/phd/migration/data/supp2_compilation_existing_catalogs.dat"
)
t_pct_b, d_pct_b = percentile_envelope(t_b, d_b, bin_size=50, percentile=75)
early_b, late_b = fit_and_split(t_pct_b, d_pct_b, split_frac=1 / 4)

# %% ---- Combined figure ----
fig, axes = plt.subplots(
    1, 2, figsize=(6.4 / 1.5 * 1.8, 4.8 / 1.5), dpi=200, layout="constrained"
)

# # %%
# axes[0].cla()
# axes[1].cla()
plot_front(
    axes[0],
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
    mask=mask_s,
    early_ann_idx=10,
    late_ann_idx=40,
)
plot_front(
    axes[1],
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

axes[0].set_ylabel("Distance [m]")
axes[0].annotate("(a)", (0, 1.03), xycoords="axes fraction", fontsize="large")
axes[1].annotate("(b)", (0, 1.03), xycoords="axes fraction", fontsize="large")

# plt.savefig(Path.cwd() / "figures" / "paper" / "field_migration.png", dpi=200)
# plt.savefig(Path.cwd() / "overleaf" / "figures_main" / "Fig5.eps")
plt.pause(0.01)
