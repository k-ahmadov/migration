from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import trapezoid

from fracinj import io

# %%
fp = Path("/home/kahmadov/phd/migration/results/3dec/ramp/m_q-1e-08.hdf5")
# fp = Path("/home/kahmadov/phd/migration/results/3dec/linear/run-q-1e-06.hdf5")
run = io.read_hdf5(fp)

o = np.argsort(run.x_fz)


# %% discharge rate at the injection point

CW, GR = 3.4, 1.618
fig = plt.figure(
    figsize=(CW, CW / GR), dpi=150, layout="constrained", clear=True, num=1
)
ax = fig.subplots()
ax.plot(run.t, run.q[:, o][:, 0], ".")
ax.plot(run.t, run.params.m_q * run.t, "-")
ax.set(
    xlabel="Time (s)",
    ylabel="Discharge rate (m2/s)",
    # xscale='log',
    # yscale='log'
)
fig.canvas.draw_idle()
plt.pause(0.01)


# %%


x, w = io.sort_fields(run.x_vert, run.w)
V = trapezoid(w, x, axis=1)
q_in = np.gradient(V, run.t)  # inlet flux, m^2/s
fig = plt.figure(
    figsize=(CW, CW / GR), dpi=150, layout="constrained", clear=True, num=1
)
ax = fig.subplots()
ax.plot(run.t, q_in, ".")
ax.plot(run.t, run.params.m_q * run.t, "-")
fig.canvas.draw_idle()
plt.pause(0.01)


# %%
CW, GR = 3.4, 1.618
fig = plt.figure(
    figsize=(CW, CW / GR), dpi=150, layout="constrained", clear=True, num=2
)
ax = fig.subplots()
ax.plot(run.x_vert, run.w[-1], ".", mfc='none')
ax.set(
    xlabel="Distance (m)",
    ylabel="Aperture (m)",
)
fig.canvas.draw_idle()
plt.pause(0.01)
