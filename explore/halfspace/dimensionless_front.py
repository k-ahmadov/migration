from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mypackages import file_io, front_analysis, math_utils, plotting, typesdefs

# %%

# ---------------------------------------------------------------------------
# Dimensionless scalings
# ---------------------------------------------------------------------------


def x_char(params: typesdefs.Parameters) -> float:
    return params.k_n * params.w_i**4 / (12 * params.mu * params.q_0)


def t_char(params: typesdefs.Parameters) -> float:
    return params.k_n * params.w_i**5 / (12 * params.mu * params.q_0**2)


def dimensionless_distance(x, params: typesdefs.Parameters):
    return np.asarray(x) / x_char(params)


def dimensionless_time(t, params: typesdefs.Parameters):
    return np.asarray(t) / t_char(params)


# %%

# ---------------------------------------------------------------------------
# Load & analyze runs
# ---------------------------------------------------------------------------

RESULT_DIR = Path.cwd() / "results" / "halfspace" / "wi-1e-05"
Q0_VALUES = np.geomspace(1e-10, 1e-6, 5)

# Early-time runs need truncated slices (front detection breaks down late);
# later runs are well-behaved over their full range.
CUSTOM_SLICE_FRACTIONS = {0: 15, 1: 5, 2: 3}  # run_index -> keep first 1/N of points


def slice_for_run(i: int, n_points: int) -> slice:
    if i in CUSTOM_SLICE_FRACTIONS:
        return slice(None, n_points // CUSTOM_SLICE_FRACTIONS[i])
    return slice(None, None)


runs = [file_io.read_halfspace(RESULT_DIR / f"q-{q0:.0e}.hdf5") for q0 in Q0_VALUES]
results = [
    front_analysis.analyze(run, stress_front=True, slc=slice_for_run(i, len(run.t)))
    for i, run in enumerate(runs)
]

result_rigid = front_analysis.analyze_rigid(
    runs[0], stress_front=True, slc=slice(None, 50)
)
result_soft = front_analysis.analyze_soft(runs[-1], stress_front=True)

# min and max time across all simulations
t_min = runs[0].t[1]
t_max = runs[-1].t[-1]
t_range = np.geomspace(
    dimensionless_time(t_min, runs[0].params),
    dimensionless_time(t_max, runs[-1].params),
    200,
)

A_ana_rigid = float(
    np.mean(
        dimensionless_distance(result_rigid.x_front[1:], runs[0].params)
        / (dimensionless_time(result_rigid.t_front[1:], runs[0].params))
        ** result_rigid.alpha_ana
    )
)

A_ana_soft = float(
    np.mean(
        dimensionless_distance(result_soft.x_front[1:], runs[-1].params)
        / (dimensionless_time(result_soft.t_front[1:], runs[-1].params))
        ** result_soft.alpha_ana
    )
)

# %%
delta_crossover = 1e-2
x_front_crossover = math_utils.crossover(
    x=t_range,
    a=A_ana_rigid * 1.6,
    alpha=result_rigid.alpha_ana,
    b=A_ana_soft * 1.6,
    beta=result_soft.alpha_ana,
    x0=1,
    delta=delta_crossover,
)


# %% fit a function that combines both power laws

fig = plt.figure(
    figsize=(6.4 / 1.3, 4.8 / 1.3), dpi=200, layout="constrained", clear=True, num=1
)
ax = fig.subplots()

for i, (run, result) in enumerate(zip(runs, results)):
    label = (
        rf"$q_0={plotting.sci_latex(run.params.q_0)}\,\mathrm{{m^2 / s}},\ \alpha = {result.alpha_emp:.2f}$",
    )
    ax.plot(
        dimensionless_time(result.t_front[1:], run.params),
        dimensionless_distance(result.x_front[1:], run.params),
        ".",
        color=f"C{i}",
        label=label,
    )
ax.plot(
    t_range,
    x_front_crossover,
    "k--",
    label=rf"Crossover function, $\delta={delta_crossover}$",
)
ax.set(
    xlabel=r"$\bar{t} = t / t_c,\ t_c = \dfrac{x_c w_i}{q_0}$",
    ylabel=r"$\bar{x}_f = x_f / x_c,\ x_c = \dfrac{k_n w_i^4}{12 \mu q_0}$",
    title="Dimensionless front migration",
    xscale="log",
    yscale="log",
)
ax.legend(frameon=False, fontsize="small", loc="upper left")
# fig.savefig(Path.cwd() / "figures" / "dimensionless-front" / "halfspace-crossover-func.png", dpi=200)
fig.canvas.draw_idle()
plt.pause(0.01)
# %%

fig = plt.figure(
    figsize=(6.4 / 1.3, 4.8 / 1.3), dpi=200, layout="constrained", clear=True, num=1
)
ax = fig.subplots()

for i, (run, result) in enumerate(zip(runs, results)):
    label = (
        rf"$q_0={plotting.sci_latex(run.params.q_0)}\,\mathrm{{m^2 / s}},\ \alpha = {result.alpha_emp:.2f}$",
    )
    ax.plot(
        dimensionless_time(result.t_front[1:], run.params),
        dimensionless_distance(result.x_front[1:], run.params),
        ".",
        color=f"C{i}",
        label=label,
    )
    ax.plot(
        dimensionless_time(result.t_front[1:], run.params),
        dimensionless_distance(result.x_empirical()[1:], run.params),
        "k--",
        label="Empirical" if i == len(runs) - 1 else "",
    )
ax.plot(
    t_range,
    A_ana_rigid * t_range**result_rigid.alpha_ana,
    "k-",
)
ax.annotate(
    text=rf"Rigid, $\alpha={result_rigid.alpha_ana}$",
    xy=(
        t_range[10],
        A_ana_rigid * t_range[10] ** result_rigid.alpha_ana,
    ),
    xytext=(20, -40),
    textcoords="offset points",
    ha="left",
    arrowprops=dict(arrowstyle="<-", shrinkA=2, shrinkB=2),
)
ax.plot(t_range, A_ana_soft * t_range**result_soft.alpha_ana, "k-", label="Analytical")
ax.annotate(
    text=rf"Soft, $\alpha={result_soft.alpha_ana}$",
    xy=(
        t_range[-10],
        A_ana_soft * t_range[-10] ** result_soft.alpha_ana,
    ),
    xytext=(-10, -50),
    textcoords="offset points",
    ha="left",
    arrowprops=dict(arrowstyle="<-", shrinkA=2, shrinkB=2),
)
ax.text(
    0.7,
    0.3,
    r"$\bar{x}_f = A \bar{t}^{\alpha}$",
    transform=ax.transAxes,
    fontsize="large",
)
ax.set(
    xlabel=r"$\bar{t} = t / t_c,\ t_c = \dfrac{x_c w_i}{q_0}$",
    ylabel=r"$\bar{x}_f = x_f / x_c,\ x_c = \dfrac{k_n w_i^4}{12 \mu q_0}$",
    title="Dimensionless front migration",
    xscale="log",
    yscale="log",
)
ax.legend(frameon=False, fontsize="small", loc="upper left")
# fig.savefig(Path.cwd() / "figures" / "dimensionless-front" / "3dec-q-5e-0-.png", dpi=200)
fig.canvas.draw_idle()
plt.pause(0.01)


# %% a plot to check single simulation

idx = -3
title = rf"$q_0={plotting.sci_latex(runs[idx].params.q_0)}\,\mathrm{{m^2/ s}},\ \alpha = {results[idx].alpha_emp:.2f}$"
fig = plt.figure(
    figsize=(6.4 / 1.5, 4.8 / 1.5), dpi=150, layout="tight", clear=True, num=1
)
ax = fig.subplots()
ax.plot(results[idx].t_front, results[idx].x_front, ".")
ax.set(xlabel="Time [s]", ylabel="Front position [m]", title=title)
fig.canvas.draw_idle()
plt.pause(0.01)
