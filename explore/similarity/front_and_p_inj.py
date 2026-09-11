import matplotlib.pyplot as plt
import scienceplots  # noqa: F401

plt.style.use(["science", "notebook", "bright"])  # pyright: ignore[reportAttributeAccessIssue]
from fracinj import analysis, io, paths, plotting

# %%
result_dir = paths.results_dir("3dec", "wi-1e-05")

q0_values = [5e-9, 5e-7, 5e-5]

runs = [io.read_hdf5(result_dir / f"q-{q_0}.hdf5") for q_0 in q0_values]

slices = [slice(None, len(runs[0].t) // 20), slice(None, None), slice(10, None)]

front_results = [
    analysis.front_rigid(runs[0], stress_front=True, slc=slices[0]),
    analysis.analyze_front(runs[1], stress_front=True, slc=slices[1]),
    analysis.front_soft(runs[2], stress_front=True, slc=slices[2]),
]

p_inj_results = [
    analysis.p_inj_rigid(runs[0]),
    analysis.analyze_p_inj(runs[1]),
    analysis.p_inj_soft(runs[-1]),
]

# %%

fig = plt.figure(
    figsize=(6.4 * 1.5, 4.8 / 1.25), dpi=100, layout="constrained", clear=True, num=1
)
ax1, ax2 = fig.subplots(1, 2)

scatter_plots = []
for run, front_result, p_inj_result in zip(runs, front_results, p_inj_results):
    label = rf"$q_0={plotting.sci_latex(run.params.q_0)}\,\mathrm{{m^2 / s}}$"
    sc = ax1.scatter(front_result.t, front_result.y, label=label)
    scatter_plots.append(sc)
    ax2.scatter(p_inj_result.t, p_inj_result.y)

ax1.plot(front_results[0].t, front_results[0].analytical(), "k")
plotting.slope_triangle(
    ax=ax1,
    x0=front_results[0].t[8],
    prefactor=front_results[0].A_ana,  # pyright: ignore
    slope=round(front_results[0].alpha_ana, 2),  # pyright: ignore
    dx_log=0.5,
)
(ana_line,) = ax2.plot(
    p_inj_results[0].t, p_inj_results[0].analytical(), "k", label="Analytical"
)
plotting.slope_triangle(
    ax=ax2,
    x0=p_inj_results[0].t[5],
    prefactor=p_inj_results[0].A_ana,  # pyright: ignore
    slope=round(p_inj_results[0].alpha_ana, 2),  # pyright: ignore
    dx_log=1,
)

ax1.plot(front_results[2].t, front_results[2].analytical(), "k")
plotting.slope_triangle(
    ax=ax1,
    x0=front_results[2].t[5],
    prefactor=front_results[2].A_ana,  # pyright: ignore
    slope=round(front_results[2].alpha_ana, 2),  # pyright: ignore
    dx_log=0.5,
)
ax2.plot(p_inj_results[2].t, p_inj_results[2].analytical(), "k")
plotting.slope_triangle(
    ax=ax2,
    x0=p_inj_results[2].t[0],
    prefactor=p_inj_results[2].A_ana,  # pyright: ignore
    slope=round(p_inj_results[2].alpha_ana, 2),  # pyright: ignore
    dx_log=1,
)

ax1.plot(front_results[1].t, front_results[1].empirical(), "k--")
plotting.slope_triangle(
    ax=ax1,
    x0=front_results[1].t[100],
    prefactor=front_results[1].A_emp,
    slope=round(front_results[1].alpha_emp, 2),
    dx_log=0.5,
)
(emp_line,) = ax2.plot(
    p_inj_results[1].t, p_inj_results[1].empirical(), "k--", label="Empirical"
)
plotting.slope_triangle(
    ax=ax2,
    x0=p_inj_results[1].t[50],
    prefactor=p_inj_results[1].A_emp,
    slope=round(p_inj_results[1].alpha_emp, 2),
    dx_log=1,
)

ax1.set_ylabel("Front position [m]")
ax2.set_ylabel("Presure at injector [Pa]")

for ax in [ax1, ax2]:
    ax.set(xscale="log", yscale="log", xlabel="Time [s]")

ax2.legend(handles=[ana_line, emp_line], loc="center left", bbox_to_anchor=(0, 0.4))
fig.legend(
    handles=scatter_plots, loc="outside lower center", ncols=3, borderaxespad=0.1
)

fig.savefig("/home/kahmadov/phd/migration/documents/2026-09-10-summary/front-and-p-inj.eps")
fig.canvas.draw_idle()
plt.pause(0.01)
