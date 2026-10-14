import matplotlib.pyplot as plt
import numpy as np

from fracinj import detection, io, paths, plotting

DAY = 86400.0

# %%

filepath = paths.results_dir("halfspace", "ramp", "long") / "m_q-1e-13.hdf5"
run = io.read_hdf5(filepath)
x, sn = io.sort_fields(run.x_sc, run.sn)

n_profiles = 30
# evenly spaced in time (skipping t ~ 0, where the profile is flat), snapped to stored samples
t_targets = np.linspace(run.t[-1] / n_profiles, run.t[-1], n_profiles)
idx = np.unique(np.searchsorted(run.t, t_targets).clip(0, len(run.t) - 1))

tension = -sn[idx] / 1e3  # tensile stress, kPa
offset = 0.3 * np.ptp(
    tension
)  # vertical spacing; < profile height, so neighbours may touch

# stress front (most tensile point) at the stacked times, from the injection point at t = 0
x_front, _ = detection.find_stress_front(x, sn, L=run.params.L)
x_front = np.concatenate(([0.0], x_front[idx]))
y_front = (np.arange(len(x_front)) - 1) * offset  # zero lines; t = 0 one spacing below

# %%

CW, GR = 3.4, 1.618
fig = plt.figure(figsize=(CW, CW), dpi=200, layout="constrained", clear=True, num=1)
ax = fig.subplots()
for k, (i, tn) in enumerate(zip(idx, tension)):
    y0 = k * offset  # earliest time at the bottom
    ax.axhline(y0, color="0.8", lw=0.4, zorder=0)
    ax.plot(x / 1000, y0 + tn, color="k", lw=0.8)
    if (len(idx) - 1 - k) % 2:  # label every other profile, always including the latest
        continue
    ax.text(
        1.01,
        y0,
        f"{run.t[i] / DAY:.1f} d",
        transform=ax.get_yaxis_transform(),
        va="center",
        fontsize="small",
    )
ax.plot(x_front / 1000, y_front, color="C0", lw=1.5, ls="-", label="Stress front")
ax.set(
    xlabel="Distance $x$ (km)",
    xlim=(0, run.params.L / 1000),
    yticks=[],
    xticks=[0, 1, 2, 3],
)
ax.text(1.01, 0.96, "Time", transform=ax.transAxes, va="bottom", fontsize="small")
ax.text(0.01, 0.95, "Stress profiles", transform=ax.transAxes, va="bottom", fontsize="small")

# scale bar for the stress amplitude
bar = 10 ** np.floor(np.log10(offset / 2))
ax.plot(
    [0.03 * run.params.L / 1000] * 2,
    [-2 * offset, -2 * offset + bar],
    color="k",
    lw=1,
)
ax.text(
    0.045 * run.params.L / 1000,
    -2 * offset + bar / 2,
    rf"{bar:g} kPa $-\sigma_n$",
    va="center",
    fontsize="small",
)
ax.legend(fontsize='small')
ax.tick_params(axis="x", top=False, which="both")
for side in ("left", "right", "top"):
    ax.spines[side].set_visible(False)
fig.canvas.draw_idle()
plt.pause(0.01)

# %%

plotting.save_figure(fig, "stress-profiles-stacked", overleaf=True)

# %%
