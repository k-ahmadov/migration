from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from fracinj import detection, io

# %%
run = io.read_halfspace(
    Path("/home/kahmadov/phd/migration/results/halfspace/radial/ramp/m_q-1e-08.hdf5")
)

x_front, idx_front = detection.find_stress_front(run.x_sc, run.sn, run.params.L)

# %%

CW, GR = 3.4, 1.618
fig = plt.figure(
    figsize=(CW, CW / GR), dpi=150, layout="constrained", clear=True, num=1
)
ax = fig.subplots()
# ax.plot(run.x_sc, run.sn[len(run.t) // 100])
ax.plot(run.x_sc, run.sn[len(run.t) // 2])
# ax.plot(run.x_sc, run.sn[-1])
ax.set(
    xlabel="x label",
    ylabel="y label",
    title="title",
)
fig.canvas.draw_idle()
plt.pause(0.01)

# %%
plt.figure(dpi=200)
plt.plot(run.t[:idx_front], x_front)
plt.show()
# %%
np.polyfit(np.log(run.t), np.log(x_front), deg=1)
