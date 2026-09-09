# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mypackages import file_io, front_analysis, front_detection, physics

# %%

result_dir = Path.cwd() / "results" / "3dec" / "runs"
run = file_io.read_run(result_dir / "run-q-1e-06.hdf5")

x_sc, p = file_io.sort_fields(run.x_sc, run.p)

# remove duplicate x positions
_, unique_idx = np.unique(x_sc, return_index=True)
x_sc_unique = x_sc[unique_idx]
p_unique = p[:, unique_idx]

t_idx = 100
dp = np.gradient(p_unique[t_idx], x_sc_unique)

# %%
fig, ax = plt.subplots()
ax.plot(x_sc, p[t_idx])
ax.plot(x_sc_unique, dp, ".")
ax.plot(x_sc_unique[np.argmin(dp)], np.min(dp), ".")
ax.set_xlabel("x label")
ax.set_ylabel("y label")
ax.set_title("title")
fig.tight_layout()
plt.show()
