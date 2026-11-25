import matplotlib.pyplot as plt
import numpy as np

from fracinj import analysis, io, math_utils, paths, physics, plotting

COLUMN_WIDTH = 3.25  # in
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
        "markevery": 0.1,
        "label": rf"${plotting.sci_latex(p.m_q)}$",
    }


# %%

result_dir = paths.results_dir("halfspace", "ramp")
mq_values = np.geomspace(1e-14, 1e-7, 8)
runs = [io.read_halfspace(result_dir / f"m_q-{m_q:.0e}.hdf5") for m_q in mq_values]

results_front = [analysis.analyze_front(run, stress_front=True) for run in runs]
front_rigid = analysis.front_rigid_ramp(runs[0], stress_front=True)
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


CW, GR = 3.25, 1.618
fig = plt.figure(
    figsize=(1.4 * CW, CW / GR),
    dpi=150,
    layout="constrained",
    clear=True,
    num=2,
)
ax = fig.subplots()

handles = []
for i, (run, rf, sc) in enumerate(zip(runs, results_front, scales)):
    tf = rf.t / sc.t_c
    (h,) = ax.plot(tf[tf > 0], rf.y[tf > 0] / sc.x_c, **marker_style(i, run.params))
    handles.append(h)

(l_cross,) = ax.plot(
    t_nd, x_front_crossover, "-", color="C0", lw=1.5, label="Crossover"
)
(l_asym,) = ax.plot(
    t_nd, rigid_front, color="tab:gray", ls=":", lw=1.2, label="Asymptotes"
)
ax.plot(t_nd, soft_front, color="tab:gray", ls=":", lw=1.2)
ax.annotate(
    r"$\eta_{\mathrm{f}}\, 2 \sqrt{ D t }$",
    xy=(t_nd[50], rigid_front[50]),
    xytext=(10, 30),
    textcoords="offset points",
    ha="right",
    arrowprops={"arrowstyle": "<-"},
)
ax.annotate(
    r"$\zeta_{\mathrm{f}} \left( a m_{\mathrm{q}}^{3} t^{7} \right)^{1/5}  $",
    xy=(t_nd[-30], soft_front[-30]),
    xytext=(-5, -50),
    textcoords="offset points",
    ha="center",
    va="top",
    arrowprops={"arrowstyle": "<-"},
)

ax.axvline(1, color="tab:gray", ls="--")
ax.annotate(
    r"$\hat{t}=1$",
    (1, 1e2),
    color="tab:gray",
    ha="center",
    backgroundcolor="1",
)
ax.set(
    xlabel=r"$\hat{t}$",
    ylabel=r"$\ell / x_{*}$",
    xscale="log",
    yscale="log",
    ylim=(x_pooled.min() / 2, x_pooled.max() * 2),
)
ax.legend(handles=[l_cross, l_asym], loc="upper left")
# one-column m_q legend right of the axes; as a figure legend, constrained
# layout sizes it to its contents, so the axes take the remaining width.
# The frame is off, so borderpad (default 0.4) only widens the gap to the axes.
leg = fig.legend(
    handles=handles,
    title=r"$m_\mathrm{q}$ ($\mathrm{m^2/s^2}$)",
    loc="outside right center",
    borderpad=1.0,
    ncols=1,
    frameon=False,
    # handletextpad=0.2,
    # labelspacing=0.3,
)

fig.canvas.draw_idle()
plt.pause(0.01)

# %%

# plotting.save_figure(fig, "ramp-front-dimensionless", overleaf=True)
