from pathlib import Path
from typing import cast

import h5py
import matplotlib.pyplot as plt
import numpy as np

from mypackages.file_io import read_run

result_dir = Path.cwd() / "results" / "fvm-elastic" / "runs"
filename = "run-q-1e-04.hdf5"
filepath = result_dir / filename

with h5py.File(str(filepath), "r") as f:
    x_el = cast(h5py.Dataset, f["coordinates/x_elastic"])[:]
    t = cast(h5py.Dataset, f["coordinates/t"])[:]
    sn_tx = cast(h5py.Dataset, f["fields/stress_normal"])[:]
    p_tx = cast(h5py.Dataset, f["fields/fluid_pressure"])[:]
    params = {
        k: cast(h5py.Dataset, f[f"parameters/{k}"])[()]
        for k in cast(h5py.Group, f["parameters"]).keys()
    }


def find_stress_front(x, sn, mesh_size: float = 1.0) -> tuple[np.ndarray, int]:
    positions = x[np.argmin(sn, axis=1)]
    boundary = positions.max() - mesh_size
    idx = int(np.argmax(positions >= boundary))
    return positions[:idx], idx


# %%
result_3dec_dir = Path.cwd() / "results" / "3dec" / "runs"
filepath_3dec = result_3dec_dir / filename
run_3dec = read_run(filepath_3dec)

# %%

x_front_fvm, idx = find_stress_front(x_el, sn_tx)
t_front_fvm = t[:idx]

x_front_3dec, idx = find_stress_front(run_3dec.x_sc, run_3dec.sn)
t_front_3dec = run_3dec.t[:idx]

# %%
plt.figure()
plt.loglog(t_front_fvm[1:], x_front_fvm[1:], ".", color="tab:gray")
plt.loglog(t_front_3dec[1:], x_front_3dec[1:], "x", color="tab:gray")
plt.show()


# %%
plt.figure()
plt.loglog(t, p_tx[:, 0], ".")
plt.loglog(run_3dec.t, run_3dec.p[:, np.argmax(run_3dec.x_sc == 0)], ".")
plt.xlabel("x label")
plt.ylabel("y label")
plt.title("title")
plt.tight_layout()
plt.show()
