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

# R = N / T

# %%
t_days_arr = np.asarray(t_days)
t_start = t_days_arr[:-BIN_SIZE:BIN_SIZE]
t_end = t_days_arr[BIN_SIZE::BIN_SIZE]
t_mid = (t_start + t_end) / 2
R = BIN_SIZE / (t_end - t_start)


# %%

plt.plot(t_mid, R, '.')
plt.show()

# %%


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

