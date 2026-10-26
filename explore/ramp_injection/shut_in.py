import matplotlib.pyplot as plt
import numpy as np

from fracinj import detection, io, paths, plotting

DAY = 86400.0
CW, GR = 3.4, 1.618

# %%

filepath = paths.results_dir("halfspace", "ramp", "shut-in-0.25tc") / "m_q-1e-13.hdf5"
run = io.read_hdf5(filepath)
t_stop = run.params.t_stop
x_p, p = io.sort_fields(run.x_vert, run.p)
x_s, sn = io.sort_fields(run.x_sc, run.sn)

n_profiles = 10
# evenly spaced in time, snapped to stored samples
t_targets = np.linspace(run.t[20], run.t[-1], n_profiles)
idx = np.unique(np.searchsorted(run.t, t_targets).clip(0, len(run.t) - 1))
colors = plt.cm.viridis(np.linspace(0, 1, len(idx)))

x_front, n_front = detection.find_stress_front(x_s, sn, L=run.params.L)
p_inj = p[:, 0]

# %% pressure and stress profiles

fig1 = plt.figure(figsize=(2 * CW, CW / GR), dpi=150, layout="constrained", num=1, clear=True)
ax_p, ax_s = fig1.subplots(1, 2, sharex=True)
for i, c in zip(idx, colors):
    ls = "-" if run.t[i] <= t_stop else "--"  # dashed: after shut-in
    label = f"{run.t[i] / DAY:.1f} d"
    ax_p.plot(x_p / 1000, p[i] / 1e6, color=c, ls=ls, lw=1, label=label)
    ax_s.plot(x_s / 1000, -sn[i] / 1e3, color=c, ls=ls, lw=1)
ax_s.axhline(0, color="0.7", lw=0.5, zorder=0)
ax_p.set(xlabel="Distance $x$ (km)", ylabel="Pressure $p$ (MPa)", xlim=(0, run.params.L / 1000))
ax_s.set(xlabel="Distance $x$ (km)", ylabel=r"Tensile stress $-\sigma_n$ (kPa)")
fig1.legend(loc="outside right", fontsize="small", title="Time", frameon=False)
fig1.canvas.draw_idle()
plt.pause(0.01)

# %% stress front and injection pressure versus time

fig2 = plt.figure(figsize=(2 * CW, CW / GR), dpi=150, layout="constrained", num=2, clear=True)
ax_f, ax_i = fig2.subplots(1, 2, sharex=True)
ax_f.plot(run.t[:n_front] / DAY, x_front / 1000, color="k", lw=1)
ax_i.plot(run.t / DAY, p_inj / 1e6, color="k", lw=1)
for ax in (ax_f, ax_i):
    ax.axvline(t_stop / DAY, color="tab:red", ls="--", lw=1)
ax_i.annotate(
    "Shut-in", (t_stop / DAY, 0.05), xycoords=("data", "axes fraction"),
    xytext=(4, 0), textcoords="offset points", color="tab:red",
)
ax_f.set(xlabel="Time $t$ (d)", ylabel="Stress front $x_f$ (km)", xlim=(0, run.t[-1] / DAY))
ax_i.set(xlabel="Time $t$ (d)", ylabel=r"Injection pressure $p_{\mathsf{inj}}$ (MPa)")
fig2.canvas.draw_idle()
plt.pause(0.01)

# %%

# plotting.save_figure(fig1, "shut-in-profiles")
# plotting.save_figure(fig2, "shut-in-front-p-inj")
