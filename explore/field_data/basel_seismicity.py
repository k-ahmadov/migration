import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mypackages import physics

# %%
FILEPATH = "/home/kahmadov/phd/migration/data/supp2_compilation_existing_catalogs.dat"
BIN_SIZE = 50
PERCENTILE = 75

# %%
df = pd.read_csv(
    FILEPATH,
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

df.replace(
    {"-.--": float("nan"), "-.-": float("nan"), r"\-{3,}": float("nan")},
    inplace=True,
    regex=True,
)

# conversion to appropiate types
df["SourceDateTime"] = pd.to_datetime(df["SourceDateTime"])
for col in ["Lat", "Lon", "Dep", "X", "Y", "Z", "Mwx", "MwGEL", "MwSED", "MLSED"]:
    df[col] = pd.to_numeric(df[col])

# filter data obtained during injection
date_mask = (df["SourceDateTime"] > pd.Timestamp(year=2006, month=12, day=2)) & (
    df["SourceDateTime"] < pd.Timestamp(year=2006, month=12, day=11)
)
# df = df.loc[date_mask]
df = df.loc[date_mask].reset_index(drop=True)

# %%

timestamps = df["SourceDateTime"]
t_days = (timestamps - timestamps.iloc[0]).dt.total_seconds() / 86400

x, y, z = df["X"], df["Y"], df["Z"]
x0, y0, z0 = x.iloc[:10].median(), y.iloc[:10].median(), z.iloc[:10].median()
distance = np.sqrt((x - x0) ** 2 + (y - y0) ** 2 + (z - z0) ** 2)

# Percentile envelope binned by event count
n_bins = len(distance) // BIN_SIZE
n_trim = n_bins * BIN_SIZE
d_bins = distance.values[:n_trim].reshape(n_bins, BIN_SIZE)
t_bins = t_days.values[:n_trim].reshape(n_bins, BIN_SIZE)

p_vals = np.percentile(d_bins, PERCENTILE, axis=1, method="nearest")
idx = np.argmin(np.abs(d_bins - p_vals[:, None]), axis=1)
t_pct = t_bins[np.arange(n_bins), idx]
d_pct = d_bins[np.arange(n_bins), idx]

# %%
idx_start = len(t_pct) // 4
t_early, d_early = t_pct[:idx_start], d_pct[:idx_start]
t_late, d_late = t_pct[idx_start:], d_pct[idx_start:]

# Fits
A_early, alpha_early = physics.fit_front_power_law(t_early, d_early)
A_late, alpha_late = physics.fit_front_power_law(t_late, d_late)

d_fit_early = A_early * t_early**alpha_early
d_fit_late = A_late * t_late**alpha_late

# %%

fig, ax = plt.subplots(figsize=(6.4 / 1.4, 4.8 / 1.4), dpi=200)
ax.plot(t_days, distance, ".", color="tab:gray", ms=3)
ax.plot(t_pct, d_pct, "ko", label=f"P{PERCENTILE} per {BIN_SIZE} events")
ax.plot(
    t_early,
    d_fit_early,
    "--",
    color="tab:blue",
    lw=3,
    label="Early-time fit",
)
ax.annotate(
    "Early-time" "\n" rf"$x_f \propto t^{{{alpha_early:.2f}}}$",
    xy=(t_early[3], d_fit_early[3]),
    xytext=(-20, 30),
    textcoords="offset points",
    ha="right",
    backgroundcolor="1",
    arrowprops=dict(arrowstyle="<-", shrinkA=2, shrinkB=2),
)
ax.plot(t_late, d_fit_late, "--", color="tab:orange", label="Late-time fit", lw=3)
ax.annotate(
    "Late-time" "\n" rf"$x_f \propto t^{{{alpha_late:.2f}}}$",
    xy=(t_late[10], d_fit_late[10]),
    xytext=(-20, 30),
    backgroundcolor="1",
    textcoords="offset points",
    ha="right",
    arrowprops=dict(arrowstyle="<-", shrinkA=2, shrinkB=2),
)
ax.set(
    xlabel="Time [days]",
    ylabel="Distance [m]",
    title="Basel 2006",
    xscale="log",
    yscale="log",
    xlim=(0.5, 10),
    ylim=(50, 1000),
)
ax.spines[["right", "top"]].set_visible(False)
ax.legend(loc="lower right", framealpha=1, edgecolor="0.5")
fig.tight_layout()
plt.show()
