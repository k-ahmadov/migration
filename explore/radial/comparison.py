from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from fracinj import io

# %%
run_3dec = io.read_hdf5(
    Path("/home/kahmadov/phd/explore/radial-diffusion/results/ramp/m_Q-1e-09.hdf5")
)

run = io.read_halfspace(
    Path("/home/kahmadov/phd/migration/results/halfspace/radial/test/m_q-4e-09.hdf5")
)


# %%
idx_3dec = 200

idx = np.searchsorted(run.t, run_3dec.t[idx_3dec])
dx = np.diff(run_3dec.x_sc)[0]

# %%


# %%

idx_front_3dec = np.argmin(run_3dec.sn, axis=1)
l_3dec = run_3dec.x_sc[idx_front_3dec]

idx_front = np.argmin(run.sn, axis=1)
l = run.x_sc[idx_front]

emp_exp_front, emp_log_pref_front = np.polyfit(
    np.log(run_3dec.t), np.log(l_3dec), deg=1
)
emp_exp_inj, emp_log_pref_inj = np.polyfit(
    np.log(run_3dec.t), np.log(run_3dec.p[:, 0]), deg=1
)


# %%

CW, GR = 3.4, 1.618
fig = plt.figure(
    figsize=(2 * CW, 2 * CW / GR), dpi=200, layout="constrained", clear=True, num=1
)
sf_top, sf_bot = fig.subfigures(2, 1)

# --- top row (your first block) ---
ax1, ax2 = sf_top.subplots(1, 2)
ax1.plot(run_3dec.x_sc, run_3dec.p[idx_3dec] / 1e6, ".", label="3DEC")
ax1.plot(run.x_sc, run.p[idx] / 1e6, label="my code")
ax1.axvspan(0, dx, color="0.9")
ax1.text(dx, 0, "mesh size", fontsize="small")
ax1.set(
    xlabel="Distance (m)",
    ylabel="Fluid pressure (MPa)",
    title=f"$t={run_3dec.t[idx_3dec]:.0f}$ s",
)
ax1.legend()

ax2.plot(run_3dec.x_sc, -run_3dec.sn[idx_3dec] / 1e6, ".", label="3DEC")
ax2.plot(run.x_sc[4:], -run.sn[idx][4:] / 1e6, label="my code")
ax2.text(dx, 0, "mesh size", fontsize="small")
ax2.axvspan(0, np.diff(run_3dec.x_sc)[0], color="0.9")
ax2.set(
    xlabel="Distance (m)",
    ylabel="Tensile Stress (MPa)",
    title=f"$t={run_3dec.t[idx_3dec]:.0f}$ s",
)
ax2.legend()

# --- bottom row (your second block) ---
ax1, ax2 = sf_bot.subplots(1, 2)  # reuse names; or call them ax3, ax4
ax1.plot(run_3dec.t, l_3dec, ".", label="3DEC")
ax1.plot(run.t, l, label="my code")
ax1.plot(
    run.t,
    np.exp(emp_log_pref_front) * run.t**emp_exp_front,
    label=rf"$\propto t^{{{emp_exp_front:.2f}}}$",
)
ax1.axhline(dx, color="0.7", linestyle="--")
ax1.text(4, dx + 0.5, "3dec mesh size", fontsize="small")
ax1.legend()
ax1.set(
    xlabel="Time (s)",
    ylabel="Front position (m)",
    xscale="log",
    yscale="log",
)

ax2.plot(run_3dec.t, run_3dec.p[:, 0] / 1e6, ".", label="3DEC")
ax2.plot(run.t, run.p[:, 0] / 1e6, label="my code")
ax2.plot(
    run.t,
    np.exp(emp_log_pref_inj) * run.t**emp_exp_inj / 1e6,
    label=rf"$\propto t^{{{emp_exp_inj:.2f}}}$",
)
ax2.set(
    xlabel="Time (s)",
    ylabel="Injector pressure (MPa)",
    xscale="log",
    yscale="log",
)
ax2.legend()

fig.canvas.draw_idle()
plt.pause(0.01)
fig.savefig("/home/kahmadov/phd/migration/figures/comparison_mycode_3dec_radial.pdf")
