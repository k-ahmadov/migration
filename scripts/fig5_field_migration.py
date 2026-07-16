from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mypackages import physics
from mypackages.front_analysis import RIGID_DIFFUSION_EXPONENT, SOFT_DIFFUSION_EXPONENT


# %%
def load_soultz(filepath):
    # load data file
    df = pd.read_csv(filepath, sep=";")
    # formatting string for date and time format
    fmt = "%d/%m/%Y %H:%M:%S.%f"
    # import time
    timestamps = pd.to_datetime(
        df["Date (local)"] + " " + df["Time (local)"], format=fmt
    )
    # cut data to injection period
    date_mask = timestamps < timestamps.iloc[0] + pd.Timedelta(days=18)
    df = df.loc[date_mask]
    timestamps = timestamps[date_mask]
    # import time in seconds
    t = (timestamps - timestamps.iloc[0]).dt.total_seconds()
    # import coordinates of events
    x, y, z = df["East (m)"], df["North (m)"], df["Depth (m)"]
    # median of first 10 events
    x0, y0, z0 = x.iloc[:10].median(), y.iloc[:10].median(), z.iloc[:10].median()
    # euclidian distance
    distance = np.sqrt((x - x0) ** 2 + (y - y0) ** 2 + (z - z0) ** 2)
    return t.values, distance.values


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
    t = (timestamps - timestamps.iloc[0]).dt.total_seconds()
    # import coordinates of events
    x, y, z = df["X"], df["Y"], df["Z"]
    # median of first 10 events
    x0, y0, z0 = x.iloc[:10].median(), y.iloc[:10].median(), z.iloc[:10].median()
    # euclidian distance
    distance = np.sqrt((x - x0) ** 2 + (y - y0) ** 2 + (z - z0) ** 2)
    return t.values, distance.values


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


def fit_and_split(t_pct, d_pct, split_frac):
    "Function to extract the early- and late-time series and fit a power law function"
    # index to separate early and late data
    idx_start = int(len(t_pct) * split_frac)
    # import early and late
    t_early, d_early = t_pct[:idx_start], d_pct[:idx_start]
    t_late, d_late = t_pct[idx_start:], d_pct[idx_start:]
    # find best fit power law prefactor and exponent to the early and late time treands
    A_e, a_e = physics.fit_front_power_law(t_early, d_early)
    A_l, a_l = physics.fit_front_power_law(t_late, d_late)
    return (t_early, A_e * t_early**a_e, a_e), (t_late, A_l * t_late**a_l, a_l)


# TODO: fit analytical models to early and late time data
def analytical_fit_and_split(t_pct, d_pct, split_frac):
    "Function to extract the early- and late-time series and fit a power law function"
    # index to separate early and late data
    idx_start = int(len(t_pct) * split_frac)
    # import early and late
    t_early, d_early = t_pct[:idx_start], d_pct[:idx_start]
    t_late, d_late = t_pct[idx_start:], d_pct[idx_start:]
    A_e = physics.fit_front_power_law_fixed_alpha(
        t_early, d_early, alpha=RIGID_DIFFUSION_EXPONENT
    )
    A_l = physics.fit_front_power_law_fixed_alpha(
        t_late, d_late, alpha=SOFT_DIFFUSION_EXPONENT
    )
    return (
        t_early,
        A_e * t_early**RIGID_DIFFUSION_EXPONENT,
        RIGID_DIFFUSION_EXPONENT,
        A_e,
    ), (t_late, A_l * t_late**SOFT_DIFFUSION_EXPONENT, SOFT_DIFFUSION_EXPONENT, A_l)


def _to_days(t_sec):
    return t_sec / 86400


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

    ax.plot(
        _to_days(t),
        distance,
        ".",
        color="tab:gray",
        ms=3,
        label="Seismic events",
    )
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
    Path.cwd() / "data" / "SSFS1993-Catalogue_Bourouis.csv",
)
t_pct_s, d_pct_s = percentile_envelope(t_s, d_s, bin_size=50, percentile=90)
early_s, late_s = analytical_fit_and_split(t_pct_s, d_pct_s, split_frac=1 / 4)

# %% ---- Basel 2006 ----
t_b, d_b = load_basel(Path.cwd() / "data" / "supp2_compilation_existing_catalogs.dat")
t_pct_b, d_pct_b = percentile_envelope(t_b, d_b, bin_size=50, percentile=75)
early_b, late_b = analytical_fit_and_split(t_pct_b, d_pct_b, split_frac=1 / 4)

# %% ---- Combined figure ----
fig = plt.figure(
    figsize=(6.4 * 1.25, 4.8 / 1.5), dpi=150, layout="constrained", clear=True, num=1
)
axes = fig.subplots(1, 2)
ax1, ax2 = axes[0], axes[1]

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

# fig.savefig(Path.cwd() / "figures" / "paper" / "field_migration.png", dpi=200)
# fig.savefig(Path.cwd() / "overleaf" / "figures_main" / "Fig5.eps")

fig.canvas.draw_idle()
plt.pause(0.01)

# %%


# TODO: fit analytical models to early and late time data
def explore_analytical_fit_and_split(t_pct, d_pct, split_frac):
    "Function to extract the early- and late-time series and fit a power law function"
    # index to separate early and late data
    idx_start = int(len(t_pct) * split_frac)
    # import early and late
    t_early, d_early = t_pct[:idx_start], d_pct[:idx_start]
    t_late, d_late = t_pct[idx_start:], d_pct[idx_start:]
    A_e = physics.fit_front_power_law_fixed_alpha(
        t_early, d_early, alpha=RIGID_DIFFUSION_EXPONENT
    )
    A_l = physics.fit_front_power_law_fixed_alpha(
        t_late, d_late, alpha=SOFT_DIFFUSION_EXPONENT
    )
    return A_e, A_l


# %%

prefactor_early, prefactor_late = explore_analytical_fit_and_split(
    t_pct_s, d_pct_s, split_frac=1 / 4
)

# %%
D = prefactor_early**2 / 4
mu = 1e-3
w_i = 1e-5
k_n = 12 * D * mu / w_i**3

print(f"k_n = {k_n / 1e9:.1f} GPa/m")

a = k_n / 12 / mu
q_0 = (prefactor_late**5 / a) ** (1 / 3)
print(f"q_0 = {q_0:.1e} m^2/s")

L = 1e3
# dimensionless deformation parameter
epsilon = w_i / (L * q_0 / a)**(1/4)
print(f"epsilon = {epsilon}")


