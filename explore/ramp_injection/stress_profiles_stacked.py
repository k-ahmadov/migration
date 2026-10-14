import matplotlib.pyplot as plt
import numpy as np

from fracinj import detection, io, paths

DAY = 86400.0

# %%

filepath = paths.results_dir("halfspace", "ramp", "long") / "m_q-1e-13.hdf5"
run = io.read_hdf5(filepath)
x, sn = io.sort_fields(run.x_sc, run.sn)

n_profiles = 20
# evenly spaced in time (skipping t ~ 0, where the profile is flat), snapped to stored samples
t_targets = np.linspace(run.t[-1] / n_profiles, run.t[-1], n_profiles)
idx = np.unique(np.searchsorted(run.t, t_targets).clip(0, len(run.t) - 1))

tension = -sn[idx] / 1e3  # tensile stress, kPa
offset = 0.5 * np.ptp(tension)  # vertical spacing; < profile height, so neighbours may touch

# stress front (most tensile point) at every stored time, until it reaches the fracture end
_, n_front = detection.find_stress_front(x, sn, L=run.params.L)
# argmin snaps to the stress grid, which makes the curve step-like; refine the
# minimum with a parabola through the node and its two neighbours
sn_f = sn[:n_front]
i_min = np.argmin(sn_f, axis=1).clip(1, len(x) - 2)
rows = np.arange(n_front)
f_l, f_c, f_r = sn_f[rows, i_min - 1], sn_f[rows, i_min], sn_f[rows, i_min + 1]
shift = 0.5 * (f_l - f_r) / (f_l - 2 * f_c + f_r)  # in grid cells, within [-0.5, 0.5]
x_front = x[i_min] + shift * (x[1] - x[0])
peak = -(f_c - 0.25 * (f_l - f_r) * shift) / 1e3  # tensile peak, kPa

# start at t = 0 from the injection point with no stress
t_front = np.concatenate(([0.0], run.t[:n_front]))
x_front = np.concatenate(([0.0], x_front))
keep = t_front <= run.t[idx[-1]]
# place it on the zero lines: profiles are evenly spaced in time, so the zero line is
# linear in t (extrapolated below the first profile for t < t[idx[0]])
y_front = np.polyval(np.polyfit(run.t[idx], np.arange(len(idx)) * offset, 1), t_front)

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
        1.01, y0, f"{run.t[i] / DAY:.1f} d", transform=ax.get_yaxis_transform(),
        va="center", fontsize="small",
    )
ax.plot(x_front[keep] / 1000, y_front[keep], color="C1", lw=1, ls="--", label="Stress front")
ax.set(
    xlabel="Distance $x$ (km)",
    xlim=(0, run.params.L /1000),
    yticks=[],
    xticks=[0, 1, 2, 3],
)
ax.text(1.01, 1.0, "Time", transform=ax.transAxes, va="bottom", fontsize="small")

# scale bar for the stress amplitude
bar = 10 ** np.floor(np.log10(offset / 2))
ax.plot([0.03 * run.params.L / 1000] * 2, [-1.5 * offset, -1.5 * offset + bar], color="k", lw=1)
ax.text(
    0.045 * run.params.L / 1000, -1.5 * offset + bar / 2, rf"{bar:g} kPa $-\sigma_n$",
    va="center", fontsize="small",
)
ax.tick_params(axis="x", top=False, which="both")
for side in ("left", "right", "top"):
    ax.spines[side].set_visible(False)
fig.canvas.draw_idle()
plt.pause(0.01)

# %%
