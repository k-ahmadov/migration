from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from fracinj import detection, io

# %%
run_3dec = io.read_halfspace(
    Path("/home/kahmadov/phd/explore/radial-diffusion/results/constant/rate/q_0-1e-08.hdf5")

)

run = io.read_halfspace(
    Path("/home/kahmadov/phd/migration/results/halfspace/radial/constant/q_0-8e-08.hdf5")

)

# %%
t = run_3dec.t[-1]
sn_3dec = run_3dec.sn[-1]
p_3dec = run_3dec.p[-1]
w_3dec = run_3dec.w[-1]

idx = np.searchsorted(run.t, 5*t)-1
p = run.p[idx]
sn = run.sn[idx]
w= run.w[idx]


# %%
CW, GR = 3.4, 1.618
fig = plt.figure(figsize=(CW, CW/GR), dpi=150, layout='constrained', clear=True, num=1)
ax = fig.subplots()
# ax.plot(run_3dec.x_sc, run_3dec.p[-1])
# ax.plot(run.x_sc, -sn)
# ax.plot(run_3dec.x_sc, -run_3dec.sn[-1]+20e6)
# ax.plot(run.x_sc, p)
# ax.plot(run_3dec.x_sc,p_3dec, '.')
ax.plot(run.x_sc, w)
ax.plot(run_3dec.x_vert,w_3dec, '.')

ax.set(
    xlabel = "x label",
    ylabel = "y label",
    title = "title",
)
fig.canvas.draw_idle()
plt.pause(0.01)
