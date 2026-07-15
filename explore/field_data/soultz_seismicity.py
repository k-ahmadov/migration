import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mypackages import physics

# %%
FILEPATH = "/home/kahmadov/phd/migration/data/SSFS1993-Catalogue_Bourouis.csv"
BIN_SIZE = 50
PERCENTILE = 90
T_MAX_DAYS = 18
D_MAX_M = 1050

df = pd.read_csv(FILEPATH, sep=";")

# %%
fmt = "%d/%m/%Y %H:%M:%S.%f"
timestamps = pd.to_datetime(df["Date (local)"] + " " + df["Time (local)"], format=fmt)
t_days = (timestamps - timestamps.iloc[0]).dt.total_seconds() / 86400

x, y, z = df["East (m)"], df["North (m)"], df["Depth (m)"]
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
# Percentile envelope slices
pmask = t_pct < T_MAX_DAYS
mask = (t_days <= T_MAX_DAYS) & (distance <= D_MAX_M)
t_pct_m = t_pct[pmask]
d_pct_m = d_pct[pmask]

idx_start = len(t_pct_m) // 5
t_early, d_early = t_pct_m[:idx_start], d_pct_m[:idx_start]
t_late, d_late = t_pct_m[idx_start:], d_pct_m[idx_start:]

# Fits
A_early, alpha_early = physics.fit_front_power_law(t_early, d_early)
A_late, alpha_late = physics.fit_front_power_law(t_late, d_late)

d_fit_early = A_early * t_early**alpha_early
d_fit_late = A_late * t_late**alpha_late


# %%  Plot
fig, ax = plt.subplots(figsize=(6.4 / 1.4, 4.8 / 1.4), dpi=200)
ax.plot(t_days[mask], distance[mask], ".", color="tab:gray", ms=3)
ax.plot(t_pct_m, d_pct_m, "ko", label=f"P{PERCENTILE} per {BIN_SIZE} events")
ax.plot(
    t_early,
    d_fit_early,
    "--",
    color="tab:blue",
    lw=3,
    label="Best fit power law",
)
ax.annotate(
    "Early-time" "\n" rf"$x_f \propto t^{{{alpha_early:.2f}}}$",
    xy=(t_early[10], d_fit_early[10]),
    xytext=(-20, 30),
    textcoords="offset points",
    ha="right",
    backgroundcolor="1",
    arrowprops=dict(arrowstyle="<-", shrinkA=2, shrinkB=2),
)
ax.plot(t_late, d_fit_late, "--", color="tab:blue", lw=3)
ax.annotate(
    "Late-time" "\n" rf"$x_f \propto t^{{{alpha_late:.2f}}}$",
    xy=(t_late[40], d_fit_late[40]),
    xytext=(-20, 30),
    backgroundcolor="1",
    textcoords="offset points",
    ha="right",
    arrowprops=dict(arrowstyle="<-", shrinkA=2, shrinkB=2),
)
ax.set(
    xlabel="Time [day]",
    ylabel="Distance [m]",
    title="Soultz 1993",
    xscale="log",
    yscale="log",
    xlim=(0.5, 20),
    ylim=(50, 1100),
)
ax.spines[["right", "top"]].set_visible(False)
ax.legend(loc="lower right", framealpha=1, edgecolor="0.5")
fig.tight_layout()
plt.show()

# %%
