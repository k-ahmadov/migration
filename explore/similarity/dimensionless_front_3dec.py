from pathlib import Path

import matplotlib.pyplot as plt

from mypackages import file_io, front_analysis, plotting, typesdefs

# %%


def dimensionless_distance(
    x: typesdefs.Vector, params: typesdefs.Parameters
) -> typesdefs.Vector:
    x_char = params.k_n * params.w_i**4 / (12 * params.mu * params.q_0)
    return x / x_char


def dimensionless_time(
    t: typesdefs.Vector, params: typesdefs.Parameters
) -> typesdefs.Vector:
    t_char = params.k_n * params.w_i**5 / (12 * params.mu * params.q_0**2)
    return t / t_char


# %%

result_dir = Path.cwd() / "results" / "3dec" / "wi-1e-05"

q0_values = [5e-9, 5e-7, 5e-5]

runs = [file_io.read_run(result_dir / f"q-{q_0}.hdf5") for q_0 in q0_values]

slices = [slice(None, len(runs[0].t) // 20), slice(None, None), slice(2, None)]

results = [
    front_analysis.analyze(run, stress_front=True, slc=slc)
    for run, slc in zip(runs, slices)
]

colors = ["tab:blue", "tab:orange", "tab:green"]

# %%

fig = plt.figure(
    figsize=(6.4 / 1.5, 4.8 / 1.5), dpi=200, layout="tight", clear=True, num=1
)
ax = fig.subplots()

for run, result, color in zip(runs, results, colors):
    label = (
        rf"$q_0={plotting.sci_latex(run.params.q_0)}\,\mathrm{{m^2 \cdot s^{{-1}} }},\ \alpha_{{emp}} = {result.alpha_emp:.2f}$",
    )
    ax.plot(
        dimensionless_time(result.t_front, run.params),
        dimensionless_distance(result.x_front, run.params),
        ".",
        color=color,
        label=label,
    )
    ax.plot(
        dimensionless_time(result.t_front, run.params),
        dimensionless_distance(result.x_empirical(), run.params),
        "k--",
    )
ax.set(
    xlabel=r"$t / t^*,\ t^* = \frac{k_n w_i^5}{12 \mu q_0^2}$",
    ylabel=r"$x_f / x^*,\ x^* = \frac{k_n w_i^4}{12 \mu q_0}$",
    title="Dimensionless front migration",
    xscale="log",
    yscale="log",
)
ax.legend(frameon=False, fontsize="x-small")
# fig.savefig(Path.cwd() / "figures" / "dimensionless-front" / "3dec-q-5e-0-.png", dpi=200)
fig.canvas.draw_idle()
plt.pause(0.01)
