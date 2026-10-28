from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from fracinj import io

# %%
# fp = Path("/home/kahmadov/phd/migration/results/3dec/ramp/m_q-1e-12.hdf5")
fp = Path("/home/kahmadov/phd/migration/results/3dec/linear/run-q-1e-06.hdf5")
run = io.read_hdf5(fp)

o = np.argsort(run.x_fz)


# %%

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
)
fig.canvas.draw_idle()
plt.pause(0.01)


# %%
CW, GR = 3.4, 1.618
fig = plt.figure(
    figsize=(CW, CW / GR), dpi=150, layout="constrained", clear=True, num=1
)
ax = fig.subplots()
ax.plot(run.x_vert, run.w[200], ".")
ax.set(
    xlabel="Distance (m)",
    ylabel="Aperture (m)",
)
fig.canvas.draw_idle()
plt.pause(0.01)
