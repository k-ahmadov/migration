from pathlib import Path

import matplotlib.pyplot as plt

from mypackages import file_io, front_analysis, plotting

# %%
run_long = file_io.read_fvm(
    Path.cwd() / "results" / "fvm-elastic" / "runs" / "run-L-500.hdf5"
)
run_short = file_io.read_fvm(
    Path.cwd() / "results" / "fvm-elastic" / "runs" / "run-q-1e-04.hdf5"
)

result_long = front_analysis.analyze(run_long, stress_front=True)
result_short = front_analysis.analyze(run_short, stress_front=True)

# %%

plt.figure(figsize=(6.4 / 1.4, 4.8 / 1.4), dpi=200)
plt.plot(
    result_short.t_front,
    result_short.x_front,
    "x",
    color="tab:gray",
    label="100 m fracture",
    markersize=4,
)
plt.plot(result_short.t_front, result_short.x_empirical(), "k-")
plotting.slope_triangle(
    ax=plt.gca(),
    x0=result_short.t_front[25],
    prefactor=result_short.A_emp,
    slope=round(result_short.alpha_emp, 2),
)
plt.plot(
    result_long.t_front,
    result_long.x_front,
    ".",
    color="tab:gray",
    label="500 m fracture",
)
plt.plot(result_long.t_front, result_long.x_empirical(), "k-")
plotting.slope_triangle(
    ax=plt.gca(),
    x0=result_long.t_front[25],
    prefactor=result_long.A_emp,
    slope=round(result_long.alpha_emp, 2),
)
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Time [s]")
plt.ylabel("Stress peak position [m]")
plt.legend()
plt.tight_layout()
plt.show()
