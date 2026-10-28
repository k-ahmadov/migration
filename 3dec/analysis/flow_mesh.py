"""Flow-mesh diagnostics in the fault plane (xy): flowknot volume, flowknot
timestep and flowzone area, one panel (row) each.

Reads ``knots.csv`` and ``zones.csv`` written by ``3dec/scripts/export_flow_mesh.py``
from ``3dec/data/flow_mesh/<TAG>/``.
"""

import matplotlib.pyplot as plt
import numpy as np

from fracinj import paths, plotting

TAG = "base"
DATA_DIR = paths.ROOT / "3dec" / "data" / "flow_mesh" / TAG

# %% load


def load(name: str) -> np.ndarray:
    """Read a CSV with a header row into a structured array."""
    return np.genfromtxt(DATA_DIR / name, delimiter=",", names=True)


knots = load("knots.csv")
zones = load("zones.csv")

# %% plot

# (data, column, colorbar label), one row each
PANELS = [
    (knots, "vol", r"Knot volume (m$^3$)"),
    (knots, "timestep", "Knot timestep (s)"),
    (zones, "area", r"Zone area (m$^2$)"),
]

fig, axes = plt.subplots(
    len(PANELS),
    1,
    figsize=(7.0, 4.6),
    dpi=150,
    sharex=True,
    layout="constrained",
    num=1,
    clear=True,
)

for ax, (data, column, label) in zip(axes, PANELS):
    sc = ax.scatter(data["x"], data["y"], c=data[column], s=6, cmap="viridis")
    fig.colorbar(sc, ax=ax, label=label, pad=0.01)
    ax.set_ylabel("$y$ (m)")
    ax.set_ylim(np.min(knots["y"]) - 0.1, np.max(knots["y"]) + 0.1)

axes[-1].set_xlabel("$x$ (m)")
plotting.annotate_panel_labels(list(axes), xy=(0.01, 0.9))

fig.canvas.draw_idle()
plt.pause(0.01)

# %% save

# plotting.save_figure(fig, f"flow-mesh-{TAG}")

# %% histogram of timesteps


# %%

data = knots["vol"]
counts, bins = np.histogram(data, bins=100)

mn = np.mean(data)
med = np.median(data)
s = np.std(data)

CW, GR = 3.4, 1.618
fig = plt.figure(
    figsize=(CW, CW / GR), dpi=150, layout="constrained", clear=True, num=1
)
ax = fig.subplots()
ax.stairs(counts, bins)
ax.axvline(mn, color="C1")
ax.axvline(med, color="C2")
# ax.axvspan(mn - 2 * s, mn + 2 * s, color="0.9")
ax.set(
    xlabel="x label",
    ylabel="y label",
    title="title",
)
fig.canvas.draw_idle()
plt.pause(0.01)
