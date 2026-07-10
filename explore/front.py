# %%
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from mypackages import file_io, front_analysis, front_detection, physics

# %% --- Plotting ---------------------------------------------------------------


def plot_front_rigid(ax, result: front_analysis.FrontResultsWithAnalytical, title: str):
    t = result.t_front
    x = result.x_front
    ax.plot(t, x, ".", color="tab:gray", label="Numerical (3DEC)")
    D = (result.A_ana / result.ζ_front) ** (1 / result.α_ana)
    ax.plot(
        t,
        result.x_analytical(),
        color="k",
        label=r"Analytical - $x_f = \zeta_f \, D^{0.5} t^{0.5}$"
        "\n"
        rf"$\zeta_f={result.ζ_front:.1f}$, $D={D:.1f}~\mathsf{{m^2/s}}$",
    )
    slope_triangle(
        ax, x0=t[len(t) // 10], prefactor=result.A_ana, slope=result.α_ana, dx_log=0.5
    )
    # ax.text(
    #     t[int(0.01 * len(t))],  # pyright: ignore[reportIndexIssue]
    #     x[int(0.6 * len(x))],  # pyright: ignore[reportIndexIssue]
    #     rf"$\zeta_f={result.ζ_front:.2f}$, $D={D:.2f}~\mathsf{{m^2/s}}$",
    # )
    ax.set(
        xlabel=r"Time $t$, [s]",
        ylabel="Front distance $x_f$, [m]",
        title=title,
        xscale="log",
        yscale="log",
    )
    ax.legend(loc="upper left")


def plot_front_soft(ax, result: front_analysis.FrontResultsWithAnalytical, title: str):
    t = result.t_front
    x = result.x_front
    ax.plot(t, x, ".", color="tab:gray", label="Numerical (3DEC)")
    ax.plot(
        t,
        result.x_analytical(),
        color="k",
        label=r"Analytical - $x_f = \zeta_f \, A \, t^{0.8}$",
    )
    ax.plot(
        t,
        result.x_empirical(),
        linestyle="--",
        color="k",
        label=f"Fit - $x_f = {result.A_emp:.1f} \\, t^{{{result.α_emp:.1f}}}$",
    )
    slope_triangle(
        ax, x0=t[len(t) // 100], prefactor=result.A_ana, slope=result.α_ana, dx_log=0.3
    )
    slope_triangle(
        ax,
        x0=t[len(t) // 3],
        prefactor=result.A_emp,
        slope=round(result.α_emp, 1),
        dx_log=0.3,
    )
    # ax.text(
    #     t[int(0.01 * len(t))],
    #     x[int(0.6 * len(x))],
    #     rf"$A=\left( \frac{{k_n q_0^3}}{{12 \mu}} \right)^{{1/5}}={result.A_ana:.2f}~\mathsf{{m/s^{{4/5}}}}$",
    # )
    # ax.text(
    #     t[int(0.01 * len(t))],
    #     x[int(0.45 * len(x))],
    #     rf"$\zeta_f={result.ζ_front:.2f}$",
    # )
    ax.set(
        xlabel=r"Time $t$, [s]",
        ylabel="Front distance $x_f$, [m]",
        title=title,
        xscale="log",
        yscale="log",
    )
    ax.legend(loc="upper left")


def _setup_rcparams():
    plt.rcParams["font.size"] = 12
    plt.rcParams["text.usetex"] = False


def _annotate_regime(ax, t, x_ana, label):
    """Annotate an analytical curve at ~25% of its length."""
    i_xy = len(t) // 4
    i_txt = len(t) // 10
    ax.annotate(
        label,
        xy=(t[i_xy], x_ana[i_xy]),
        xytext=(t[i_txt], x_ana[i_txt] / 3),
        wrap=True,
        arrowprops=dict(arrowstyle="->", shrinkA=8, shrinkB=8),
    )


def slope_triangle(ax, x0, prefactor, slope, dx_log=0.3, label_slope=True):
    """Annotate a slope triangle on an existing loglog axis."""

    y0 = prefactor * x0**slope  # assumes curve passes through here; adjust as needed
    x1 = x0 * 10**dx_log
    y2 = y0 * 10 ** (slope * dx_log)

    tri = mpatches.Polygon(
        [[x0, y0], [x1, y0], [x1, y2]],
        fill=False,
        edgecolor="k",
    )
    ax.add_patch(tri)
    ax.annotate(
        "1",
        xy=((x0 * x1) ** 0.5, y0),
        xytext=(0, -12),
        textcoords="offset points",
        ha="center",
        fontsize=10,
    )
    if label_slope:
        ax.annotate(
            str(slope),
            xy=(x1, (y0 * y2) ** 0.5),
            xytext=(6, 0),
            textcoords="offset points",
            ha="left",
            fontsize=10,
        )


def plot_early_late(ax, general, early, late, run, pc):
    t_c = physics.characteristic_time(run.params)

    ax.plot(
        general.t_front,
        general.x_front,
        ".",
        color="tab:gray",
        label="Numerical (3DEC)",
    )

    ax.plot(
        early.t_front,
        early.x_analytical(),
        "--",
        color="tab:blue",
        lw=3,
        label="Analytical",
    )
    _annotate_regime(
        ax,
        early.t_front,
        early.x_analytical(),
        r"$x_f=\zeta_f D^{1/2} t^{1/2}$" + "\n(early-time)",
    )

    ax.plot(late.t_front, late.x_analytical(), "--", color="tab:blue", lw=3)
    _annotate_regime(
        ax,
        late.t_front,
        late.x_analytical(),
        r"$x_f=\zeta_f M^{1/5} q_0^{3/5} t^{4/5}$" + "\n(late-time)",
    )

    ax.axvline(t_c, color="k", ls=":", label="Characteristic time")
    ax.set(
        xscale="log",
        yscale="log",
        xlabel=r"Time $t$, [s]",
        ylabel=r"Front distance $x_f$, [m]",
        title=rf"$q_0={run.params['q_0']:.0e}~\mathsf{{m^2/s}}$, $p_c={pc / 1e6:.1f}~\mathsf{{MPa}}$",
    )
    ax.legend()
    plt.show()


# %% --- Main: rigid vs soft ---------------------------------------------------


def main_rigid_soft():
    result_dir = Path.cwd() / "results" / "3dec" / "runs"
    # figure_dir = Path.cwd() / "figures"

    run_rigid = file_io.read_run(result_dir / "run-q-1e-06.hdf5")
    rigid = front_analysis.analyze_rigid(
        run_rigid,
        front_detection.self_similar_front_threshold(
            run_rigid, θ_front=0.1, is_pressure=True
        ),
    )

    run_soft = file_io.read_run(result_dir / "run-q-1e-03.hdf5")
    soft = front_analysis.analyze_soft(
        run_soft, front_detection.constant_pressure_threshold(run_soft.t, pc=5e5)
    )

    _setup_rcparams()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.4 * 1.8, 4.8))
    plot_front_rigid(
        ax1,
        rigid,
        title=rf"Rigid case, $q_0={run_rigid.params['q_0']}~\mathsf{{m^2/s}}$",
    )
    plot_front_soft(
        ax2,
        soft,
        title=rf"Soft case, $q_0={run_soft.params['q_0']}~\mathsf{{m^2/s}}$",
    )
    # fig.savefig(figure_dir / "front.png", dpi=200)
    fig.tight_layout()
    plt.show()


# %% --- Main: early vs late time regime ---------------------------------------


def main_early_late(filename: str, pc: float):
    result_dir = Path.cwd() / "results" / "3dec" / "runs"
    run = file_io.read_run(result_dir / filename)

    general = front_analysis.analyze(
        run, front_detection.constant_pressure_threshold(run.t, pc)
    )
    early = front_analysis.analyze_early_time(run, pc)
    late = front_analysis.analyze_late_time(run, pc)

    _setup_rcparams()
    fig, ax = plt.subplots(dpi=200)
    plot_early_late(ax, general, early, late, run, pc)
    fig.tight_layout()
    plt.show()


# %% --- Run -------------------------------------------------------------------

# main_rigid_soft()
# main_early_late("run-q-1e-03.hdf5", pc=5e5)
# main_early_late("run-q-1e-04.hdf5", pc=5e5)


# %%

result_dir = Path.cwd() / "results" / "3dec" / "runs"
run_0 = file_io.read_run(result_dir / "run-q-1e-06.hdf5")
run_1 = file_io.read_run(result_dir / "run-q-1e-05.hdf5")
run_2 = file_io.read_run(result_dir / "run-q-1e-04.hdf5")
run_3 = file_io.read_run(result_dir / "run-q-1e-03.hdf5")

results_0 = front_analysis.analyze_rigid(
    run_0, threshold=front_detection.self_similar_front_threshold(run_0, θ_front=0.1)
)
results_1 = front_analysis.analyze(
    run_1, threshold=front_detection.self_similar_front_threshold(run_1, θ_front=0.1)
)
results_2 = front_analysis.analyze(
    # run_2, threshold=front_detection.self_similar_front_threshold(run_2, θ_front=0.1)
    run_2,
    threshold=front_detection.constant_pressure_threshold(run_2.t, pc=5e5),
)
results_3 = front_analysis.analyze(
    # run_3, threshold=front_detection.self_similar_front_threshold(run_3, θ_front=0.1)
    run_3,
    threshold=front_detection.constant_pressure_threshold(run_3.t, pc=5e5),
)

# %% -- plotting all the runs in one plot ----------
plt.rcParams["font.size"] = 12
fig, ax = plt.subplots(dpi=200)
ax.loglog(
    results_0.t_front,
    results_0.x_front,
    ".",
    color="tab:gray",
    label="Numerical (3DEC)",
)
ax.loglog(
    results_0.t_front, results_0.x_analytical(), "--", color="k", label="Best fit"
)
slope_triangle(
    ax,
    results_0.t_front[len(results_0.t_front) // 5],
    prefactor=results_0.A_ana,
    slope=results_0.α_ana,
)
ax.annotate(
    text=f"$q_0={run_0.params['q_0']:.0e}~\\mathsf{{m^2/s}}$",
    xy=(
        results_0.t_front[len(results_0.t_front) // 20],
        results_0.x_front[len(results_0.t_front) // 20],
    ),
    xytext=(25, 0),
    textcoords="offset points",
    ha="left",
    fontsize=10,
)
# ax.loglog(results_1.t_front, results_1.x_front, ".", color="tab:gray")
ax.loglog(results_2.t_front, results_2.x_front, ".", color="tab:gray")
ax.loglog(results_2.t_front, results_2.x_empirical(), "--", color="k")
slope_triangle(
    ax,
    results_2.t_front[len(results_2.t_front) // 5],
    prefactor=results_2.A_emp,
    slope=round(results_2.α_emp, 1),
)
ax.annotate(
    text=f"$q_0={run_2.params['q_0']:.0e}~\\mathsf{{m^2/s}}$",
    xy=(
        results_2.t_front[len(results_2.t_front) // 20],
        results_2.x_front[len(results_2.t_front) // 20],
    ),
    xytext=(10, -50),
    textcoords="offset points",
    ha="left",
    fontsize=10,
    arrowprops=dict(arrowstyle="<-", shrinkA=4, shrinkB=4),
)
ax.loglog(results_3.t_front, results_3.x_front, ".", color="tab:gray")
ax.loglog(results_3.t_front, results_3.x_empirical(), "--", color="k")
slope_triangle(
    ax,
    results_3.t_front[len(results_3.t_front) // 5],
    prefactor=results_3.A_emp,
    slope=round(results_3.α_emp, 1),
)
ax.annotate(
    text=f"$q_0={run_3.params['q_0']:.0e}~\\mathsf{{m^2/s}}$",
    xy=(
        results_3.t_front[len(results_3.t_front) // 5],
        results_3.x_front[len(results_3.t_front) // 5],
    ),
    xytext=(-20, 0),
    textcoords="offset points",
    ha="right",
    fontsize=10,
)
ax.set_xlabel("Time $t$, [s]")
ax.set_ylabel("Front position $x_f$, [m]")
ax.set_title("Constant flow rate injection")
ax.legend()
fig.tight_layout()
plt.show()


# %% --- Local exponent analysis ------------------------------------------------------------------

result_dir = Path.cwd() / "results" / "3dec" / "runs"
run = file_io.read_run(result_dir / "run-q-1e-03.hdf5")
threshold_ss = front_detection.self_similar_front_threshold(
    run, θ_front=0.1, is_pressure=True
)
threshold_cp = front_detection.constant_pressure_threshold(run.t[slice(None)], pc=1e5)
results = front_analysis.analyze(run, threshold_cp)

t, local_exponent = results.calculate_local_exponent(window=50)
# %%

fig, ax = plt.subplots()
ax.plot(t, local_exponent, ".")
ax.set_xlabel("Time [s]")
ax.set_ylabel("Local exponent")
ax.set_xscale("log")
ax.set_title(rf"$q_0={run.params['q_0']:.0e}~\mathsf{{m^2/s}}$")
fig.tight_layout()
plt.show()


# %% --- linearly increasing injection rate -----------

result_dir = Path.cwd() / "results" / "3dec" / "runs-linear"
run = file_io.read_run(result_dir / "run-q-1e-06.hdf5")
threshold_cp = front_detection.constant_pressure_threshold(run.t[slice(None)], pc=1e5)
results = front_analysis.analyze(run, threshold_cp)

# %%
fig, ax = plt.subplots()
ax.plot(results.t_front, results.x_front)
ax.set_xlabel("Time [s]")
ax.set_ylabel("Front position [m]")
ax.set_title(f"Linearly increasing injection rate with slope $q={run.params['q']}$")
fig.tight_layout()
plt.show()


# %% -- constant pressure results -----

result_dir = Path.cwd() / "results" / "3dec" / "constant-pressure"

run_rigid = file_io.read_pickle(result_dir / "rigid.pkl")


threshold_cp = front_detection.constant_pressure_threshold(
    run_rigid.t[slice(None)], pc=1e5
)
rigid = front_analysis.analyze_rigid(run_rigid, threshold_cp)

run_soft = file_io.read_pickle(result_dir / "soft.pkl")
threshold_cp = front_detection.constant_pressure_threshold(run_soft.t, pc=1e5)
soft = front_analysis.analyze_rigid(run_soft, threshold_cp)


# %% --- Plot rigid and soft for constant pressure ---------------------------------------------------------------
plt.rcParams["font.size"] = 12
fig, ax = plt.subplots(dpi=200)

for results, run, idx_frac, xytext in [
    (rigid, run_rigid, 12, (-15, 0)),
    (soft, run_soft, 100, (85, 20)),
]:
    D = (results.A_ana / results.ζ_front) ** (1 / results.α_ana)
    idx = len(results.t_front) // idx_frac

    ax.loglog(results.t_front, results.x_front, ".", color="tab:gray")
    ax.loglog(results.t_front, results.x_analytical(), "-", color="k")
    slope_triangle(
        ax,
        x0=results.t_front[idx],
        prefactor=results.A_ana,
        slope=results.α_ana,
        dx_log=0.3,
    )
    ax.annotate(
        text=rf"$x_f = \zeta_f \, \sqrt{{D t}}$"
        "\n"
        rf"$\zeta_f={results.ζ_front:.1f}$, $D={D:.1f}~\mathsf{{m^2/s}}$"
        "\n"
        rf"$k_n={run.params['k_n'] / 1e9}~\mathsf{{GPa/m}}$",
        xy=(results.t_front[idx], results.x_front[idx]),
        xytext=xytext,
        textcoords="offset points",
        ha="right" if xytext[0] < 0 else "left",
        fontsize=10,
    )

ax.get_lines()[0].set_label("Numerical (3DEC)")
ax.get_lines()[1].set_label("Analytical")
ax.legend()
ax.set_xlabel("Time $t$, [s]")
ax.set_ylabel("Front position $x_f$, [m]")
ax.set_title("Fluid pressure front migration")
fig.tight_layout()
plt.show()


# %% -- low initial aperture results -----

result_dir = Path.cwd() / "results" / "3dec" / "runs-wi-1e-05"
run = file_io.read_run(result_dir / "run-q-1e-09.hdf5")

x_front, idx_cut = front_detection.find_stress_front(run.x_sc, run.sn, mesh_size=2)
t_front = run.t[:idx_cut]
A_emp, α_emp = physics.fit_front_power_law(
    t_front[: len(t_front) // 4], x_front[: len(t_front) // 4]
)
# front_analysis.analyze_rigid()
# %%
plt.figure()
plt.loglog(t_front, x_front, ".")
plt.loglog(t_front, A_emp * t_front**α_emp, ".")
plt.ylabel("Front position [m]")
plt.xlabel("Time [s]")
plt.title("title")
plt.tight_layout()
plt.show()

# %%
np.polyfit(t_front, x_front, deg=1)

# %% -- stress front ------

# %%


result_dir = Path.cwd() / "results" / "3dec" / "constant-pressure"

run = file_io.read_pickle(result_dir / "soft.pkl")
x_front, idx_cut = front_detection.find_stress_front(run.x_sc, run.sn, mesh_size=1)
t_front = run.t[:idx_cut]
idx = len(t_front)
A_emp, α_emp = physics.fit_front_power_law(
    t_front[:idx],
    x_front[:idx],
)
# %%
plt.figure()
plt.loglog(t_front, x_front, ".")
# plt.loglog(t_front, A_ana * t_front**α_ana, "-")
plt.loglog(t_front, A_emp * t_front**α_emp, "--")
plt.ylabel("Front position [m]")
plt.xlabel("Time [s]")
plt.title("title")
plt.tight_layout()
plt.show()
