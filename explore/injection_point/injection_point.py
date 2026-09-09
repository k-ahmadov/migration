import importlib
from pathlib import Path

import matplotlib.pyplot as plt

from mypackages import file_io, p_inj_analysis, physics

importlib.reload(p_inj_analysis)

# %%

result_dir = Path.cwd() / "results" / "3dec" / "stress-bc"
result_fvm_dir = Path.cwd() / "results" / "fvm-elastic" / "wi-1e-05"
fvm_results = False
if fvm_results:
    run = file_io.read_halfspace(result_fvm_dir / "q-5e-07.hdf5")
else:
    run = file_io.read_run(result_dir / "q-5e-07-fine.hdf5")

result = p_inj_analysis.analyze(run)
result_early = p_inj_analysis.analyze_early(run)
result_late = p_inj_analysis.analyze_late(run)

# %%

fig = plt.figure(
    figsize=(6.4 / 1.5, 4.8 / 1.5), dpi=150, layout="tight", clear=True, num=1
)
ax = fig.subplots()
ax.plot(result.t, result.p_inj_num, ".", color="tab:gray", label="3DEC")
ax.plot(result_early.t, result_early.p_inj_fit_ana(), "k-", label="Early")
ax.plot(result_late.t, result_late.p_inj_fit_ana() , "k--", label="Late")
ax.set(
    xlabel="Time [s]",
    ylabel="Pressure [Pa]",
    title="Injection point",
    xscale="log",
    yscale="log",
)
ax.axvline(
    physics.critical_time(run.params),
    color="k",
    ls="-.",
    label=rf"$t_c ={physics.critical_time(run.params):.0f}\,\mathrm{{s}}$ (ana.)",
)
ax.legend()
fig.canvas.draw_idle()
plt.pause(0.01)

# %%


#
#
# result = front_analysis.analyze(run, stress_front=True, slc=slice(None, None))
#
# result_soft = front_analysis.analyze_soft(run, stress_front=True)
# result_rigid = front_analysis.analyze_rigid(run, stress_front=True)
# idx_crossover = 40
# result_early = front_analysis.analyze_early_time(
#     run, stress_front=True, slc=slice(0, idx_crossover)
# )
# result_late = front_analysis.analyze_late_time(
#     run, stress_front=True, slc=slice(idx_crossover, None)
# )
