from pathlib import Path

import matplotlib.pyplot as plt

from fracinj import analysis, io

# %%
# fp = Path("/home/kahmadov/phd/migration/results/halfspace/ramp/m_q-1e-08.hdf5")
fp = Path("/home/kahmadov/phd/migration/results/3dec/ramp/m_q-1e-08.hdf5")

# run = io.read_halfspace(fp)
run = io.read_hdf5(fp)

result = analysis.analyze_front(run, stress_front=True)

# %%

CW, GR = 3.4, 1.618
fig = plt.figure(
    figsize=(CW, CW / GR), dpi=200, layout="constrained", clear=True, num=1
)
ax = fig.subplots()
ax.plot(result.t, result.y, '.', mfc='none')
ax.set(
    xlabel="Time (s)",
    ylabel="Distance (m)",
    # xscale='log',
    # yscale='log'
)
fig.canvas.draw_idle()
plt.pause(0.01)
