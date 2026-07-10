from pathlib import Path

import matplotlib.pyplot as plt

# import numpy as np
from mypackages import file_io, front_analysis, physics

# %%
result_dir = Path.cwd() / "results" / "3dec" / "runs-wi-1e-05"
run = file_io.read_run(result_dir / "run-q-5e-05.hdf5")
result = front_analysis.analyze(run, stress_front=True, slc=slice(None, None))

result_soft = front_analysis.analyze_soft(run, stress_front=True)
result_rigid = front_analysis.analyze_rigid(run, stress_front=True)
result_early = front_analysis.analyze_early_time(
    run, stress_front=True, slc=slice(5, 50)
)

# %% --- Plotting -------------------

fig, ax = plt.subplots(
    1,
    1,
    figsize=(6.4 / 1.75, 4.8 / 1.75),
    dpi=200,
    sharey=True,
    constrained_layout=True,
)
ax.plot(result.t_front, result.x_front, ".", color="tab:gray", label="3DEC")
# ax.plot(result.t_front, result.x_empirical(), "--", color="k", label="Fit")
# ax.plot(
#     result_early.t_front,
#     result_early.x_analytical(),
#     "-",
#     color="k",
#     label="early ana.",
# )
ax.plot(
    result_rigid.t_front,
    result_rigid.x_analytical(),
    "-",
    color="k",
    label="Rigid ana.",
)
ax.plot(
    result_soft.t_front, result_soft.x_analytical(), "--", color="k", label="Soft ana."
)
ax.set(
    xlabel="Time [s]",
    ylabel="Distance [m]",
    title="Tensile peak migration",
    xscale="log",
    yscale="log",
)
ax.legend()
fig.tight_layout()
plt.show()


# %%
print(f"Scaling exponent: α = {result.α_emp:.2f}")

print("w_0/w_i =", run.w[len(result_soft.t_front)][run.x_vert == 0][0] / run.params.w_i)

print(
    "θ_∞ = (t_c/t)^(1/5) =",
    (physics.characteristic_time(run.params) / result.t_front[-1]) ** (1 / 5),
)
