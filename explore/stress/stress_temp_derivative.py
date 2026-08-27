from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mypackages import file_io, plotting

# %%
result_dir = Path.cwd() / "results" / "3dec" / "runs"
run_soft = file_io.read_run(result_dir / "run-q-1e-03.hdf5")
run_rigid = file_io.read_run(result_dir / "run-q-1e-06.hdf5")

# %%


def compute_dtau_dt(run, stop=-1, mu=0.6):
    assert run.tau is not None
    tau_c = run.tau[:stop] - mu * (run.sn[:stop] - run.p[:stop])
    _, tau_c = file_io.sort_fields(run.x_sc, tau_c)
    dtau_c = np.gradient(tau_c.T, run.t[:stop], axis=1)
    return dtau_c


# %%
dtau_c_soft = compute_dtau_dt(run_soft)
dtau_c_rigid = compute_dtau_dt(run_rigid)
# %%

fig, (ax1, ax2) = plt.subplots(
    1, 2, figsize=(6.4, 4.8 / 1.8), constrained_layout=True, dpi=200
)
im1 = ax1.imshow(
    dtau_c_soft,
    aspect="auto",
    origin="lower",
    extent=(
        run_soft.t.min(),
        run_soft.t.max(),
        run_soft.x_sc.min(),
        run_soft.x_sc.max(),
    ),
    vmin=np.percentile(dtau_c_soft, 5),
    vmax=np.percentile(dtau_c_soft, 95),
)
ax1.set(
    xlabel="Time [s]",
    ylabel="Distance [m]",
    # xscale='log',
    # yscale='log',
)
fig.colorbar(im1, ax=ax1, format="%.0e")
im2 = ax2.imshow(
    dtau_c_rigid,
    aspect="auto",
    origin="lower",
    extent=(
        run_rigid.t.min(),
        run_rigid.t.max(),
        run_rigid.x_sc.min(),
        run_rigid.x_sc.max(),
    ),
    vmin=np.percentile(dtau_c_rigid, 5),
    vmax=np.percentile(dtau_c_rigid, 95),
)
ax2.set(
    xlabel="Time [s]",
    ylabel="Distance [m]",
    # xscale='log',
    # yscale='log',
)
fig.colorbar(im2, ax=ax2, format="%.1e")
fig.suptitle("Coulomb Stress Rate")
plt.show()


# %%
t_i = 100
idx_x = np.argmax(run_soft.x_sc == 100)

# %%
