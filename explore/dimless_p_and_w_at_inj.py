from pathlib import Path

import matplotlib.pyplot as plt

from mypackages import file_io, physics

# %%

result_dir = Path.cwd() / "results" / "3dec" / "runs-wi-1e-05"

q_values = [5e-5, 5e-7, 5e-9]

runs = {q: file_io.read_run(result_dir / f"run-q-{q:.0e}.hdf5") for q in q_values}

p_inj_dict = {q: run.p[:, 0] for q, run in runs.items()}

w_inj_dict = {q: run.w[:, 0] - run.params.w_i for q, run in runs.items()}

char = {q: physics.dimensionalize(run.params) for q, run in runs.items()}

# %%

fig = plt.figure(
    figsize=(6.4 / 1.5, 4.8 / 1.5), dpi=150, layout="tight", clear=True, num=1
)
ax = fig.subplots()
for q, p_inj in p_inj_dict.items():
    ax.plot(runs[q].t / char[q][1], p_inj, ".", color="tab:gray")
ax.set(
    xlabel="Time [s]",
    ylabel="Pressure at the injection point [Pa]",
    title="3 simulations",
    xscale="log",
    yscale="log",
)
plt.show()

# %%

fig = plt.figure(
    figsize=(6.4 / 1.5, 4.8 / 1.5), dpi=150, layout="tight", clear=True, num=1
)
ax = fig.subplots()
for q, w_inj in w_inj_dict.items():
    ax.plot(runs[q].t / char[q][1], w_inj / char[q][0], ".", color="tab:gray")
ax.set(
    xlabel=r"$t/t^*$",
    ylabel=r"$(w_0-w_i)/w^*$",
    title="3 simulations",
    xscale="log",
    yscale="log",
)
plt.show()
