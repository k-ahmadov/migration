from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scienceplots  # noqa: F401

plt.style.use(["science", "notebook"])  # pyright: ignore[reportAttributeAccessIssue]
from fracinj import analysis, detection, io, paths, physics, plotting

# %%

result_dir = paths.results_dir("3dec", "constant-pressure")

run_rigid = io.read_pickle(result_dir / "rigid.pkl")

threshold_cp = detection.constant_pressure_threshold(run_rigid.t, pc=1e5)
rigid = analysis.front_rigid(run_rigid, threshold=threshold_cp)

run_soft = io.read_pickle(result_dir / "soft.pkl")
threshold_cp = detection.constant_pressure_threshold(run_soft.t, pc=1e5)
soft = analysis.front_soft(run_soft, threshold=threshold_cp)

mask = soft.t > 0
D = physics.diffusivity(run_soft.params)
alpha_ana = 0.5
scale = (D * soft.t[mask]) ** alpha_ana
amp_unit = D**0.5
zeta = float(np.mean(soft.y[mask] / scale))
A_ana = zeta * amp_unit
soft_analytical = A_ana * soft.t ** (alpha_ana)

colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

# %%

fig = plt.figure(
    figsize=(6.4 / 1.5, 4.8 / 1.5), dpi=150, layout="constrained", num=1, clear=True
)
ax = fig.subplots()
ax.scatter(rigid.t, rigid.y, s=10, label="Numerical", color=colors[-1])
ax.plot(rigid.t, rigid.analytical(), label="Analytical", color=colors[-2])
ax.annotate(
    rf"$k_n={run_rigid.params.k_n / 1e9}~\mathsf{{GPa/m}}$",
    xy=(rigid.t[10], rigid.y[10]),
    xytext=(0, 10),
    textcoords="offset points",
    ha="right",
)
plotting.slope_triangle(ax, rigid.t[5], rigid.A_ana, rigid.alpha_ana)


ax.scatter(soft.t, soft.y, s=10, color=colors[-1])
ax.plot(soft.t, soft_analytical, color=colors[-2])
ax.annotate(
    rf"$k_n={run_soft.params.k_n / 1e9}~\mathsf{{GPa/m}}$",
    xy=(soft.t[80], soft.y[80]),
    xytext=(30, 0),
    textcoords="offset points",
    ha="left",
)
plotting.slope_triangle(ax, soft.t[25], A_ana, alpha_ana)
ax.set(
    xlabel="Time [s]",
    ylabel="Front position [m]",
    title="Constant pressure injection",
    xscale="log",
    yscale="log",
)
ax.legend()
fig.savefig("/home/kahmadov/phd/migration/documents/2026-09-10-summary/figures/const-pressure-front.eps")
fig.canvas.draw_idle()
plt.pause(0.01)
