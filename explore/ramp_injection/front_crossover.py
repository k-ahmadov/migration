import matplotlib.pyplot as plt
import numpy as np

from fracinj import analysis, io, math_utils, paths, physics, plotting

COLUMN_WIDTH = 3.4  # in
MARKERS = ["o", "s", "^", "D", "v", "p", "<", ">"]

# %%


def marker_style(i, p):
    """Marker style for run i, matching dimensionless.py."""
    return {
        "marker": MARKERS[i % len(MARKERS)],
        "mfc": "k" if i % 2 == 0 else "none",
        "mec": "k",
        "mew": 0.7,
        "ms": 3.5,
        "ls": "none",
        "markevery": 0.05,
        "label": rf"${plotting.sci_latex(p.m_q)}$",
    }


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
front_rigid = analysis.front_rigid_ramp(runs[0], stress_front=True, slc=slices[0])
front_soft = analysis.front_soft_ramp(runs[-1], stress_front=True)

# pool every run onto dimensionless axes (t / t_c, x_f / x_c)
scales = [physics.ramp_rate_fixed_aperture_cubic(run.params) for run in runs]
t_pooled = np.concatenate([res.t / sc.t_c for res, sc in zip(results_front, scales)])
x_pooled = np.concatenate([res.y / sc.x_c for res, sc in zip(results_front, scales)])
x_pooled = x_pooled[t_pooled > 0]
t_nd = np.geomspace(t_pooled[t_pooled > 0].min(), t_pooled.max(), 200)

rigid_front = front_rigid.analytical(t_nd * scales[0].t_c) / scales[0].x_c
soft_front = front_soft.analytical(t_nd * scales[-1].t_c) / scales[-1].x_c

x_front_crossover = math_utils.crossover(
    x=t_nd,
    a=front_rigid.A_ana * 15,
    alpha=front_rigid.alpha_ana,
    b=front_soft.A_ana * 53,
    beta=front_soft.alpha_ana,
    x0=1,
    delta=5e-2,
)

# %%

fig = plt.figure(
    figsize=(1.5 * COLUMN_WIDTH, 0.675 * COLUMN_WIDTH),
    dpi=150,
    layout="constrained",
    clear=True,
    num=2,
)
axes = fig.subplot_mosaic(
    [["front", "legend"], ["front", "alpha"]],
    width_ratios=[2.2, 1],
    height_ratios=[1, 1],
)
ax = axes["front"]

handles = []
for i, (run, rf, sc) in enumerate(zip(runs, results_front, scales)):
    tf = rf.t / sc.t_c
    (h,) = ax.plot(tf[tf > 0], rf.y[tf > 0] / sc.x_c, **marker_style(i, run.params))
    handles.append(h)

(l_cross,) = ax.plot(
    t_nd, x_front_crossover, "-", color="C0", lw=1.2, label="Crossover"
)
(l_asym,) = ax.plot(
    t_nd, rigid_front, color="tab:gray", ls=":", lw=1, label="Asymptotes"
)
ax.plot(t_nd, soft_front, color="tab:gray", ls=":", lw=1)
ax.annotate(
    r"Rigid, $\zeta_{\mathsf{f}} \sqrt{ D t }$",
    xy=(t_nd[50], rigid_front[50]),
    xytext=(10, 30),
    textcoords="offset points",
    ha="right",
    arrowprops={"arrowstyle": "<-"},
)
ax.annotate(
    r"Soft, $\zeta_{\mathsf{f}} \left( a m_{\mathsf{q}}^{3} t^{7} \right)^{1/5}  $",
    xy=(t_nd[-30], soft_front[-30]),
    xytext=(-10, -50),
    textcoords="offset points",
    ha="center",
    va="top",
    arrowprops={"arrowstyle": "<-"},
)

ax.axvline(1, color="tab:gray", ls="--")
ax.annotate(
    r"$\hat{t}_{\mathsf{c}}=1$",
    (1, 1e2),
    color="tab:gray",
    ha="center",
    backgroundcolor="1",
)
ax.set(
    xlabel=r"Dimensionless time $\hat{t}$",
    ylabel=r"Dimensionless front $\hat{x}_{\mathsf{f}}$",
    xscale="log",
    yscale="log",
    ylim=(x_pooled.min() / 2, x_pooled.max() * 2),
)
ax.legend(handles=[l_cross, l_asym], loc="upper left")
axes["legend"].axis("off")
leg = axes["legend"].legend(
    handles=handles,
    title=r"$m_\mathsf{q}$ ($\mathsf{m^2/s^2}$)",
    loc="center right",
    ncols=2,
    frameon=False,
    columnspacing=0.8,
    handletextpad=0.2,
)

# empirical time exponent of each run, between the rigid and soft limits
ax_a = axes["alpha"]
for i, (run, rf) in enumerate(zip(runs, results_front)):
    ax_a.plot(
        run.params.m_q,
        rf.alpha_emp,
        **(marker_style(i, run.params) | {"markevery": None}),
    )
for fit in (front_rigid, front_soft):
    ax_a.axhline(fit.alpha_ana, color="tab:gray", ls=":", lw=1)
# linear in log10(m_q), since the pane's x-axis is logarithmic
mq = np.array([run.params.m_q for run in runs])
alpha_emp = np.array([rf.alpha_emp for rf in results_front])
slope, intercept = np.polyfit(np.log10(mq), alpha_emp, 1)
mq_fit = np.array([3e-15, 3e-7])
ax_a.plot(mq_fit, slope * np.log10(mq_fit) + intercept, color="C6", ls="--", lw=1, zorder=0)
ax_a.set(
    xscale="log",
    xlabel=r"$m_\mathsf{q}$ ($\mathsf{m^2/s^2}$)",
    ylabel=r"Exponent $\alpha$",
    xlim=(3e-15, 3e-7),
    ylim=(0.35, 1.55),
)

fig.canvas.draw_idle()
plt.pause(0.01)

# %%

# plotting.save_figure(fig, "ramp-front-dimensionless", overleaf=True)
