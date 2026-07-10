from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mypackages import file_io, front_analysis, plotting

# %%
result_dir = Path.cwd() / "results" / "fvm-elastic" / "for-V-T" / "mixed"
filenames = [
    "run-kn-1.0e+10.hdf5",
    "run-kn-1.0e+13.hdf5",
    "run-kn-1.1e+12.hdf5",
    "run-kn-2.2e+12.hdf5",
    "run-kn-3.3e+12.hdf5",
    "run-kn-4.4e+12.hdf5",
    "run-kn-5.6e+12.hdf5",
    "run-kn-6.7e+12.hdf5",
    "run-kn-7.8e+12.hdf5",
    "run-kn-8.9e+12.hdf5",
]

runs = [file_io.read_fvm(result_dir / filename) for filename in filenames]

# %%
results = [front_analysis.analyze(run, stress_front=True) for run in runs]

duration = [result.t_front[-1] - result.t_front[0] for result in results]
velocity = [result.calculate_velocity() for result in results]

exponent, prefactor = np.polyfit(np.log(duration), np.log(velocity), deg=1)

# %%
plt.ion()
fig = plt.figure(
    figsize=(6.4 / 1.3, 4.8 / 1.5), dpi=150, layout="tight", clear=True, num=1
)
# %%
fig.clf()
ax = fig.add_subplot(111)
sc = ax.scatter(
    duration,
    velocity,
    c=[run.params.k_n / 1e9 for run in runs],
    cmap="viridis",
    marker="x",
    label="FVM",
)
ax.plot(duration, np.exp(prefactor) * duration**exponent, "k-", color="k", label="Fit")
plotting.slope_triangle(
    ax, duration[3], np.exp(prefactor), round(exponent, 2), dx_log=0.3, inverse=True
)
ax.set(
    xlabel="Duration [s]",
    ylabel="Velocity [m/s]",
    title="Velocity vs Duration scaling",
    xscale="log",
    yscale="log",
)
cbar = fig.colorbar(sc, ax=ax)
cbar.set_label(r"$k_n$, [GPa/m]")
ax.legend()
plt.pause(0.01)

# %% plot front migration
fig, ax = plt.subplots(
    figsize=(6.5 / 1.5, 4.8 / 1.5), dpi=140, clear=True, layout="tight"
)
ax.cla()
for idx in range(0, len(results), 1):
    (m1,) = ax.plot(results[idx].t_front, results[idx].x_front, ".", color="tab:gray")
    # (l1,) = ax.plot(
    #     results[idx].t_front, velocity[idx] * results[idx].t_front, "-", color="k"
    # )
ax.set(
    xlabel="Time [s]", ylabel="Distance [m]", title="title", xscale="log", yscale="log"
)
# ax.legend(handles=(m1, l1), labels=("3DEC", "Velocity est."))  # pyright: ignore[reportPossiblyUnboundVariable]
plt.pause(0.01)


# %% plot pressure profiles
ax.cla()
idx = 9
plotting.plot_profiles(
    ax, runs[idx].x_sc, runs[idx].p, runs[idx].t, cut=len(runs[idx].t) // 2
)
plt.pause(0.01)
print(runs[idx].params.k_n / 1e9)

# %% V-T for square root migration

diffusivity = 1e-1
prefactor = 4 * np.pi * diffusivity
duration = np.geomspace(1e3, 1e7, 10)
distance = (prefactor * duration) ** (0.5)
velocity = distance / duration
exponent, prefactor = np.polyfit(np.log(duration), np.log(velocity), deg=1)

# %%
plt.ion()
fig = plt.figure(
    figsize=(6.4 / 1.5, 4.8 / 1.5), dpi=130, layout="tight", clear=True, num=1
)
fig.clf()
ax = fig.add_subplot(111)
# %%
ax.cla()
ax.plot(duration, velocity, ".", color="tab:gray", label=r"From $x_f \propto t^{0.5}$ ")
ax.plot(duration, np.exp(prefactor) * duration**exponent, "k--", label="Fit")
plotting.slope_triangle(
    ax, duration[3], np.exp(prefactor), round(exponent, 2), dx_log=0.5, inverse=True
)
ax.set(
    xlabel="Duration [s]",
    ylabel="Velocity [m/s]",
    title="Velocity vs Duration for rigid case",
    xscale="log",
    yscale="log",
)
ax.legend()
plt.pause(0.01)

# %%
diffusivity = 1e-1
prefactor = 4 * np.pi * diffusivity
duration = np.geomspace(1e3, 1e7, 10)
distance = (prefactor * duration) ** (0.8)
velocity = distance / duration
exponent, prefactor = np.polyfit(np.log(duration), np.log(velocity), deg=1)

# %%
ax.cla()
ax.plot(duration, velocity, ".", color="tab:gray", label=r"From $x_f \propto t^{0.8}$")
ax.plot(duration, np.exp(prefactor) * duration**exponent, "k--", label="Fit")
plotting.slope_triangle(
    ax, duration[3], np.exp(prefactor), round(exponent, 2), dx_log=0.5, inverse=True
)
ax.set(
    xlabel="Duration [s]",
    ylabel="Velocity [m/s]",
    title="Velocity vs Duration for soft case",
    xscale="log",
    yscale="log",
)
ax.legend()
plt.pause(0.01)
