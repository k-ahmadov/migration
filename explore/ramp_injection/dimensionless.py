import matplotlib.pyplot as plt
import numpy as np

from fracinj import analysis, io, paths, physics, plotting

COLUMN_WIDTH = 3.4  # in
MARKERS = ["o", "s", "^", "D", "v", "p", "<", ">"]

# %%


def marker_style(i, p):
    """Marker style for run i, so every run looks the same in both panels."""
    return {
        "marker": MARKERS[i % len(MARKERS)],
        "mfc": "k" if i % 2 == 0 else "none",
        "mec": "k",
        "mew": 0.7,
        "ms": 3.5,
        "ls": "none",
        "markevery": 0.05,
        "label": rf"$m_\mathsf{{q}}={plotting.sci_latex(p.m_q)}\,\mathsf{{m^2/s}}$",
    }


def annotate_tc(ax: plt.Axes, y: float) -> None:
    ax.axvline(1, color="tab:gray", ls="--")
    ax.annotate(
        r"$\hat{t}_{\mathsf{c}}=1$",
        (1, y),
        color="tab:gray",
        ha="center",
        backgroundcolor="1",
    )


# %%

result_dir = paths.results_dir("halfspace", "ramp")
mq_values = np.geomspace(1e-14, 1e-7, 8)
runs = [io.read_halfspace(result_dir / f"m_q-{m_q:.0e}.hdf5") for m_q in mq_values]

# early runs: keep only the early part (late-time front detection is noisy)
keep_fraction = {0: 1 / 5, 1: 1 / 2, 2: 1 / 2}
slices = [
    slice(int(len(run.t) * keep_fraction.get(i, 1.0))) for i, run in enumerate(runs)
]

results_front = [
    analysis.analyze_front(run, stress_front=True, slc=slc)
    for run, slc in zip(runs, slices)
]
results_p_inj = [analysis.analyze_p_inj(run, slc=slc) for run, slc in zip(runs, slices)]
front_rigid = analysis.front_rigid_ramp(runs[0], stress_front=True, slc=slices[0])
front_soft = analysis.front_soft_ramp(runs[-1], stress_front=True)
p_inj_rigid = analysis.p_inj_rigid_ramp(runs[0], slc=slices[0])
p_inj_soft = analysis.p_inj_soft_ramp(runs[-1])

# pool every run onto dimensionless axes (t / t_c, x_f / x_c, p / k_n w_c)
scales = [physics.ramp_rate_fixed_aperture_cubic(run.params) for run in runs]
t_nd = np.concatenate([res.t / sc.t_c for res, sc in zip(results_front, scales)])
t_nd = np.geomspace(t_nd[t_nd > 0].min(), t_nd.max(), 200)


def analytical_nd(fit, sc, y_c):
    """Analytical curve of ``fit`` on the pooled dimensionless time."""
    return fit.analytical(t_nd * sc.t_c) / y_c


p_c = [run.params.k_n * sc.w_c for run, sc in zip(runs, scales)]
rigid_p_inj = analytical_nd(p_inj_rigid, scales[0], p_c[0])
soft_p_inj = analytical_nd(p_inj_soft, scales[-1], p_c[-1])
rigid_front = analytical_nd(front_rigid, scales[0], scales[0].x_c)
soft_front = analytical_nd(front_soft, scales[-1], scales[-1].x_c)

# %%

fig = plt.figure(
    figsize=(2 * COLUMN_WIDTH, 1.25 * 0.675 * COLUMN_WIDTH),
    dpi=150,
    layout="constrained",
    clear=True,
    num=1,
)
ax1, ax2 = fig.subplots(1, 2, sharex=True)

handles = []
for i, (run, rf, rp, sc) in enumerate(zip(runs, results_front, results_p_inj, scales)):
    style = marker_style(i, run.params)
    tp, tf = rp.t / sc.t_c, rf.t / sc.t_c
    ax1.plot(tp[tp > 0], rp.y[tp > 0] / p_c[i], **style)
    (h,) = ax2.plot(tf[tf > 0], rf.y[tf > 0] / sc.x_c, **style)
    handles.append(h)
fig.legend(handles=handles, loc="outside upper center", ncols=4, frameon=False)

ax1.plot(t_nd[:-30], rigid_p_inj[:-30], "k-", lw=1.2)
plotting.slope_triangle(
    ax1, x0=t_nd[20], y0_data=rigid_p_inj[20], slope=p_inj_rigid.alpha_ana, dx_log=0.7
)
ax1.annotate(
    r"Rigid, $\theta_0 k_{\mathsf{n}} m_{\mathsf{q}} \sqrt{ t^{3}/D }$",
    xy=(t_nd[50], rigid_p_inj[50]),
    xytext=(40, 45),
    textcoords="offset points",
    ha="right",
    arrowprops={"arrowstyle": "<-"},
)

ax1.plot(t_nd, soft_p_inj, "k-", lw=1.2)
plotting.slope_triangle(
    ax1,
    x0=t_nd[-30],
    y0_data=soft_p_inj[-30],
    slope=p_inj_soft.alpha_ana,
    dx_log=0.7,
    inverse=True,
)
ax1.annotate(
    r"Soft, $\theta_0 k_{\mathsf{n}} \left( m_{\mathsf{q}}^{2} t^{3} / a \right)^{1/5}$",
    xy=(t_nd[-50], soft_p_inj[-50]),
    xytext=(0, -40),
    textcoords="offset points",
    ha="center",
    va="top",
    arrowprops={"arrowstyle": "<-"},
)
annotate_tc(ax=ax1, y=5e-4)

(l_ana,) = ax2.plot(t_nd, rigid_front, "k-", lw=1.2, label="Analytical")
plotting.slope_triangle(
    ax2, x0=t_nd[10], y0_data=rigid_front[10], slope=front_rigid.alpha_ana, dx_log=0.7
)
ax2.annotate(
    r"Rigid, $\zeta_{\mathsf{f}} \sqrt{ D t }$",
    xy=(t_nd[50], rigid_front[50]),
    xytext=(10, 30),
    textcoords="offset points",
    ha="right",
    arrowprops={"arrowstyle": "<-"},
)

ax2.plot(t_nd[70:], soft_front[70:], "k-", lw=1.2)
plotting.slope_triangle(
    ax2,
    x0=t_nd[-50],
    y0_data=soft_front[-50],
    slope=front_soft.alpha_ana,
    dx_log=0.7,
    inverse=True,
)
ax2.annotate(
    r"Soft, $\zeta_{\mathsf{f}} \left( a m_{\mathsf{q}}^{3} t^{7} \right)^{1/5}  $",
    xy=(t_nd[-30], soft_front[-30]),
    xytext=(-10, -50),
    textcoords="offset points",
    ha="center",
    va="top",
    arrowprops={"arrowstyle": "<-"},
)
annotate_tc(ax=ax2, y=1e2)

for ax in (ax1, ax2):
    ax.set(xlabel=r"Dimensionless time $\hat{t}$", xscale="log", yscale="log")
    ax.legend(handles=[l_ana])
ax1.set(ylabel=r"Dimensionless pressure $\hat{p}$")
ax2.set(ylabel=r"Dimensionless front $\hat{x}_{\mathsf{f}}$")

fig.canvas.draw_idle()
plt.pause(0.01)

# %%

# plotting.save_figure(fig, "ramp-inj-dimensionless")
