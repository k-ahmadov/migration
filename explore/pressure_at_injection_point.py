import importlib
from pathlib import Path

import matplotlib.pyplot as plt

from mypackages import file_io, p_inj_analysis, plotting

importlib.reload(p_inj_analysis)

# %%

result_dir = Path.cwd() / "results" / "3dec" / "runs-wi-1e-05"
run = file_io.read_run(result_dir / "run-q-5e-05.hdf5")

result_soft = p_inj_analysis.analyze_soft(run)

# %%
fig, ax = plt.subplots()
ax.plot(run.t, result_soft.p_inj_num, ".", color="tab:gray", label="3DEC")
ax.plot(
    run.t,
    result_soft.p_inj_analytical(run.t) / 1.06,
    "-",
    color="k",
    label="Soft ana.",
)
plotting.slope_triangle(
    ax, x0=run.t[100], prefactor=result_soft.A_ana / 1.06, slope=result_soft.α_ana
)
ax.set(
    xlabel="Time [s]",
    ylabel="Pressure at the injection point [Pa]",
    title="Pressure at the injection point",
    xscale="log",
    yscale="log",
)
ax.legend()
fig.tight_layout()
plt.show()

# %%

# %%

# result_dir = Path.cwd() / "results" / "3dec" / "runs"
# run_rigid = file_io.read_run(result_dir / "run-q-1e-06.hdf5")
# rigid = p_inj_analysis.analyze_rigid(run_rigid)
# run_soft = file_io.read_run(result_dir / "run-q-1e-03.hdf5")
# soft = panalyze_soft(run_soft)
#
# # %%
# plt.rcParams["font.size"] = 14
# fig, ax = plt.subplots(dpi=150, num=1, clear=True)
# plot_rigid(ax, rigid, run_rigid)
# plt.show()
#
# # %%
#
# fig, ax = plt.subplots(dpi=150, num=1, clear=True)
# plot_soft(ax, soft, run_soft)
# plt.show()
#
# # %%
#
# run_intermedieate = file_io.read_run(result_dir / "run-q-1e-04.hdf5")
# rigid_interm = analyze_rigid(run_intermedieate)
# soft_interm = analyze_soft(run_intermedieate)
#
# # %%
# plt.rcParams["font.size"] = 12
# fig, ax = plt.subplots(dpi=150, num=1, clear=True)
# ax.loglog(
#     rigid_interm["t"],
#     rigid_interm["p_0"],
#     ".",
#     color="tab:gray",
#     label="Numerical (3DEC)",
# )
# ax.loglog(
#     rigid_interm["t"],
#     rigid_interm["p_0_ana"] / 1.3,
#     color="tab:blue",
#     label="Analytical rigid \n(prefactor adjusted)",
# )
#
# ax.loglog(
#     soft_interm["t"],
#     soft_interm["p_0_ana"] / 1.6,
#     "--",
#     color="tab:blue",
#     label="Analytical soft \n(prefactor adjusted)",
# )
# ax.set_xlabel("Time [s]")
# ax.set_ylabel("Pressure at injection point [Pa]")
# ax.set_title(f"$q_0={run_intermedieate.params['q_0']:.0e}~\\mathrm{{m^2/s}}$")
# ax.legend()
#
# plt.show()
#
# # fig.savefig("/home/kahmadov/phd/migration/figures/p-at-inj/q-1e-06.png")
