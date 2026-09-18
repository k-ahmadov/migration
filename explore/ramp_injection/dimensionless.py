import matplotlib.pyplot as plt
import numpy as np
import scienceplots  # noqa: F401

from fracinj import analysis, io, math_utils, paths, physics, plotting

plt.style.use(["science", "bright", "latex-sans"])  # pyright: ignore[reportAttributeAccessIssue]
plt.rcParams["font.family"] = "sans-serif"


COLUMN_WIDTH = 3.25  # in

# %%


def annotate_tc(ax: plt.Axes, y: float) -> None:
    ax.axvline(1, color="tab:gray", ls="--")
    ax.annotate(
        r"$\hat{t}_{\mathrm{c}}=1$",
        (1, y),
        color="tab:gray",
        ha="center",
        backgroundcolor="1",
    )


MARKERS = ["o", "s", "^", "D", "v", "p", "<", ">"]


def plot_numerical_results(ax1, ax2, params, results_front, results_p_inj, scales):
    handles = []
    for i, (p, rf, rp, sc) in enumerate(
        zip(params, results_front, results_p_inj, scales)
    ):
        style = {
            "marker": MARKERS[i % len(MARKERS)],
            "mfc": "k" if i % 2 == 0 else "none",
            "mec": "k",
            "mew": 0.7,
            "ms": 3.5,
            "ls": "none",
            "label": rf"$m_\mathrm{{q}}={plotting.sci_latex(p.m_q)}\,\mathrm{{m^2/s}}$",
        }
        tp, yp = rp.t / sc.t_c, rp.y / (sc.w_c * p.k_n)
        tf, yf = rf.t / sc.t_c, rf.y / sc.x_c
        ax1.plot(tp[tp > 0], yp[tp > 0], markevery=0.05, **style)
        (h,) = ax2.plot(tf[tf > 0], yf[tf > 0], markevery=0.05, **style)
        handles.append(h)
    ax2.figure.legend(
        handles=handles, loc="outside upper center", ncols=4, frameon=False
    )


# %%

result_dir = paths.results_dir("halfspace", "ramp")
mq_values = np.geomspace(1e-14, 1e-7, 8)
runs = [io.read_halfspace(result_dir / f"m_q-{m_q:.0e}.hdf5") for m_q in mq_values]
bc = "ramp_rate"

# early runs: keep only the early part (late-time front detection is noisy)
keep_fraction = {0: 1 / 5, 1: 1 / 2, 2: 1 / 2}
results_front = [
    analysis.analyze_front(
        run,
        stress_front=True,
        slc=slice(int(len(run.t) * keep_fraction.get(i, 1.0))),
    )
    for i, run in enumerate(runs)
]

result_front_rigid = analysis.front_rigid_ramp(
    runs[0], stress_front=True, slc=slice(int(len(runs[0].t) * keep_fraction[0]))
)
result_front_soft = analysis.front_soft_ramp(runs[-1], stress_front=True)

results_p_inj = [
    analysis.analyze_p_inj(run, slc=slice(int(len(run.t) * keep_fraction.get(i, 1.0))))
    for i, run in enumerate(runs)
]

result_p_inj_rigid = analysis.p_inj_rigid_ramp(
    runs[0], slc=slice(int(len(runs[0].t) * keep_fraction[0]))
)
result_p_inj_soft = analysis.p_inj_soft_ramp(runs[-1])

# pool every run onto dimensionless axes (t / t_c, x_f / x_c)
scales = [physics.ramp_rate_fixed_aperture_cubic(run.params) for run in runs]
tc = [scale.t_c for scale in scales]
xc = [scale.x_c for scale in scales]
pc = [run.params.k_n * scale.w_c for run, scale in zip(runs, scales)]

t_nd = np.concatenate([res.t / t for res, t in zip(results_front, tc)])
x_nd = np.concatenate([res.y / x for res, x in zip(results_front, xc)])
positive = t_nd > 0
t_nd, x_nd = t_nd[positive], x_nd[positive]
t_range_nd = np.geomspace(t_nd.min(), t_nd.max(), 200)
rigid_p_inj = result_p_inj_rigid.analytical(t_range_nd * tc[0]) / pc[0] / 1.3
soft_p_inj = result_p_inj_soft.analytical(t_range_nd * tc[-1]) / pc[-1] / 1.2
rigid_front = result_front_rigid.analytical(t_range_nd * tc[0]) / xc[0]
soft_front = result_front_soft.analytical(t_range_nd * tc[-1]) / xc[-1]

# %%

fig = plt.figure(
    figsize=(2 * COLUMN_WIDTH, 1.25 * 0.675 * COLUMN_WIDTH),
    dpi=150,
    layout="constrained",
    clear=True,
    num=1,
)
ax1, ax2 = fig.subplots(1, 2, sharex=True)

plot_numerical_results(
    ax1,
    ax2,
    params=[run.params for run in runs],
    results_front=results_front,
    results_p_inj=results_p_inj,
    scales=scales,
)


ax1.plot(t_range_nd[:-30], rigid_p_inj[:-30], "k-", lw=1.2)
plotting.slope_triangle(
    ax1,
    x0=t_range_nd[20],
    y0_data=rigid_p_inj[20],
    slope=result_p_inj_rigid.alpha_ana,
    dx_log=0.7,
)
ax1.annotate(
    r"Rigid, $\theta_0 k_{\mathrm{n}} m_{\mathrm{q}} \sqrt{ t^{3}/D }$",
    xy=(t_range_nd[50], rigid_p_inj[50]),
    xytext=(40, 45),
    textcoords="offset points",
    ha="right",
    arrowprops={"arrowstyle": "<-"},
)

ax1.plot(t_range_nd, soft_p_inj, "k-", lw=1.2)
plotting.slope_triangle(
    ax1,
    x0=t_range_nd[-30],
    y0_data=soft_p_inj[-30],
    slope=result_p_inj_soft.alpha_ana,
    dx_log=0.7,
    inverse=True,
)
ax1.annotate(
    r"Soft, $\theta_0 k_{\mathrm{n}} \left( m_{\mathrm{q}}^{2} t^{3} / a \right)^{1/5}$",
    xy=(t_range_nd[-50], soft_p_inj[-50]),
    xytext=(0, -40),
    textcoords="offset points",
    ha="center",
    va="top",
    arrowprops={"arrowstyle": "<-"},
)
annotate_tc(ax=ax1, y=5e-4)

(l1,) = ax2.plot(t_range_nd, rigid_front, "k-", lw=1.2, label="Analytical")
plotting.slope_triangle(
    ax2,
    x0=t_range_nd[10],
    y0_data=rigid_front[10],
    slope=result_front_rigid.alpha_ana,
    dx_log=0.7,
)
ax2.annotate(
    r"Rigid, $\zeta_{\mathrm{f}} \sqrt{ D t }$",
    xy=(t_range_nd[50], rigid_front[50]),
    xytext=(10, 30),
    textcoords="offset points",
    ha="right",
    arrowprops={"arrowstyle": "<-"},
)

ax2.plot(t_range_nd[70:], soft_front[70:], "k-", lw=1.2)
plotting.slope_triangle(
    ax2,
    x0=t_range_nd[-50],
    y0_data=soft_front[-50],
    slope=result_front_soft.alpha_ana,
    dx_log=0.7,
    inverse=True,
)
ax2.annotate(
    r"Soft, $\zeta_{\mathrm{f}} \left( a m_{\mathrm{q}}^{3} t^{7} \right)^{1/5}  $",
    xy=(t_range_nd[-30], soft_front[-30]),
    xytext=(-10, -50),
    textcoords="offset points",
    ha="center",
    va="top",
    arrowprops={"arrowstyle": "<-"},
)
annotate_tc(ax=ax2, y=1e2)

for ax in [ax1, ax2]:
    ax.set(
        # xlabel=r"$t/t_{\mathrm{c}},\ t_{\mathrm{c}} = \left( \dfrac{a w_{\mathrm{i}}^{5}}{m_{\mathrm{q}}^{2}} \right)^{1/3} $",
        xlabel=r"Dimensionless time $\hat{t}$",
        xscale="log",
        yscale="log",
    )
ax1.set(
    # ylabel=r"$p_{0} / p_{\mathrm{c}}, \ p_{c}= k_{\mathrm{n}} w_{\mathrm{i}}$",
    ylabel=r"Dimensionless pressure $\hat{p}$",
)
ax2.set(
    # ylabel=r"$x / x_{\mathrm{c}},\ x_{\mathrm{c}} = \left( \dfrac{a^{2} w_{\mathrm{i}}^{7}}{m_{\mathrm{q}}} \right)^{1/3} $"
    ylabel=r"Dimensionless front $\hat{x}_{\mathrm{f}}$"
)


ax1.legend(handles=[l1])
ax2.legend(handles=[l1])

# fig.savefig("/home/kahmadov/phd/migration/figures/paper/ramp-inj-dimensionles.eps")

fig.canvas.draw_idle()
plt.pause(0.01)

# %%

delta_crossover = 5e-2
x_front_crossover = math_utils.crossover(
    x=t_range_nd,
    a=result_front_rigid.A_ana * 15,
    alpha=result_front_rigid.alpha_ana,
    b=result_front_soft.A_ana * 53,
    beta=result_front_soft.alpha_ana,
    x0=1,
    delta=delta_crossover,
)


# %%


def marker_style(i, p):
    """Marker style for run i, shared by all figures so runs look identical."""
    return {
        "marker": MARKERS[i % len(MARKERS)],
        "mfc": "k" if i % 2 == 0 else "none",
        "mec": "k",
        "mew": 0.7,
        "ms": 3.5,
        "ls": "none",
        "label": rf"$m_\mathrm{{q}}={plotting.sci_latex(p.m_q)}\,\mathrm{{m^2/s}}$",
    }


def plot_front_data(ax, params, results_front, scales):
    for i, (rf, sc) in enumerate(zip(results_front, scales)):
        tf, yf = rf.t / sc.t_c, rf.y / sc.x_c
        ax.plot(tf[tf > 0], yf[tf > 0], "o", mfc="none", mec="k", mew=0.6, ms=3,
                markevery=0.05, label="Numerical" if i == 0 else None)


# %%

fig = plt.figure(
    figsize=(COLUMN_WIDTH, 0.675 * COLUMN_WIDTH),
    dpi=150,
    layout="constrained",
    clear=True,
    num=2,
)
ax = fig.subplots()

handles = plot_front_data(
    ax,
    params=[run.params for run in runs],
    results_front=results_front,
    scales=scales,
)

(l_cross,) = ax.plot(t_range_nd, x_front_crossover, "-", color="tab:blue", lw=1.2, label="Crossover")
(l_asym,) = ax.plot(
    t_range_nd, rigid_front, color="tab:gray", ls=":", lw=1, label="Asymptotes"
)
ax.plot(t_range_nd, soft_front, color="tab:gray", ls=":", lw=1)

ax.set(
    xlabel=r"Dimensionless time $\hat{t}$",
    ylabel=r"Dimensionless front $\hat{x}_{\mathrm{f}}$",
    xscale="log",
    yscale="log",
    ylim=(x_nd.min() / 2, x_nd.max() * 2),
)
annotate_tc(ax=ax, y=1e2)
ax.legend(loc="upper left")


fig.canvas.draw_idle()
plt.pause(0.01)
# fig.savefig("/home/kahmadov/phd/migration/figures/paper/ramp-inj-crossover.eps")
# fig.savefig("/home/kahmadov/phd/migration/tex/2026-09-10_summary/figures/ramp-inj-crossover.eps")
