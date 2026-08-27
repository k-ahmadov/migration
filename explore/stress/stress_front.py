from pathlib import Path

import matplotlib.pyplot as plt

# import numpy as np
from mypackages import file_io, front_analysis, physics, plotting

# %%
result_dir = Path.cwd() / "results" / "3dec" / "stress-bc"
result_fvm_dir = Path.cwd() / "results" / "fvm-elastic" / "wi-1e-05"
fvm_results = True
if fvm_results:
    run = file_io.read_halfspace(result_fvm_dir / "q-5e-07.hdf5")
else:
    run = file_io.read_run(result_dir / "q-5e-07-fine.hdf5")

result = front_analysis.analyze(run, stress_front=True, slc=slice(None, None))

result_soft = front_analysis.analyze_soft(run, stress_front=True)
result_rigid = front_analysis.analyze_rigid(run, stress_front=True)
idx_crossover = 40
result_early = front_analysis.analyze_early_time(
    run, stress_front=True, slc=slice(0, idx_crossover)
)
result_late = front_analysis.analyze_late_time(
    run, stress_front=True, slc=slice(idx_crossover, None)
)

# %% --- Plotting -------------------
plot_early_late = True
plot_fit = False
fig, ax = plt.subplots(
    1, 1, figsize=(6.4 / 1.5, 4.8 / 1.5), dpi=150, layout="constrained"
)
ax.plot(result.t_front, result.x_front, ".", color="tab:gray", label="Halspace" if fvm_results else "3DEC")
if plot_fit:
    ax.plot(result.t_front, result.x_empirical(), ":", color="k", label="Fit")
if plot_early_late:
    ax.plot(
        result_early.t_front,
        result_early.x_analytical(),
        "-",
        color="k",
        label="Early ana.",
    )
    ax.plot(
        result_late.t_front,
        result_late.x_analytical(),
        "--",
        color="k",
        label="Late ana.",
    )
else:
    ax.plot(
        result_rigid.t_front,
        result_rigid.x_analytical(),
        "-",
        color="k",
        label="Rigid ana.",
    )
    ax.plot(
        result_soft.t_front,
        result_soft.x_analytical(),
        "--",
        color="k",
        label="Soft ana.",
    )
ax.set(
    xlabel="Time [s]",
    ylabel="Distance [m]",
    title=rf"$q_0={plotting.sci_latex(run.params.q_0)}\,\mathrm{{m^2 \cdot s^{{-1}} }}$",
    xscale="log",
    yscale="log",
)
ax.axvline(
    run.t[idx_crossover],
    color="k",
    ls=":",
    label=rf"$t_c ={run.t[idx_crossover]:.0f}\,\mathrm{{s}}$ (emp.)",
)
ax.axvline(
    physics.critical_time(run.params),
    color="k",
    ls="-.",
    label=rf"$t_c ={physics.critical_time(run.params):.0f}\,\mathrm{{s}}$ (ana.)",
)
ax.legend()

plt.show()


# %%
print(f"Scaling exponent: alpha = {result.alpha_emp:.2f}")

# print("w_0/w_i =", run.w[len(result_soft.t_front)][run.x_vert == 0][0] / run.params.w_i)

print(
    "theta_∞ = (t_c/t)^(1/5) =",
    (physics.critical_time(run.params) / result.t_front[-1]) ** (1 / 5),
)

epsilon = run.params.w_i / (run.params.L * run.params.q_0 / physics.parameter_a(run.params))**(1/4)
print(f"epsilon = {epsilon}")

t_char = (run.params.L**5 / physics.parameter_a(run.params)/ run.params.q_0**3)**(1/4)
print(f"t* = { t_char }")

print(run.t[-1])


print((physics.diffusivity(run.params) * physics.critical_time(run.params) )**(1/2))
