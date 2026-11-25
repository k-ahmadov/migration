import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LogNorm

from fracinj import io, paths, physics
from fracinj.plotting import annotate_panel_labels
from fracinj.solvers import closed_form, similarity

M_TO_MICRON = 1e6
S_TO_DAY = 1 / 86400
# %%

filepath = paths.results_dir("halfspace", "radial", "ramp") / "m_q-1e-10.hdf5"
run = io.read_halfspace(filepath)
p = run.params

# %%

n_t = len(run.t)
t_window = (0, 1.0)
start = max(int(t_window[0] * n_t), 1)  # skip t = 0 (log scale)
stop = int(t_window[1] * n_t)

# log-spaced in *time*, snapped to nearest available sample
n_profiles = 12
t_targets = np.geomspace(run.t[start], run.t[stop - 1], n_profiles)
idx = np.unique(np.searchsorted(run.t, t_targets).clip(start, stop - 1))

t_days = run.t * S_TO_DAY  # for the colour scale only
norm = LogNorm(vmin=float(t_days[idx[0]]), vmax=float(t_days[idx[-1]]))
cmap = "viridis_r"
colors = mpl.colormaps[cmap](norm(t_days[idx]))

# %% similarity scalings
#
# Radial ramp: -r a w^n w_r -> m_pq t  as r -> 0,  m_pq = m_q / (2 pi).
#
# Rigid (n = 0, a = D):  w - w_i = (m_pq t / D) phi(eta),    eta  = r / sqrt(D t)
# Soft  (n = 3, a = k):  w       = A t^alpha theta(zeta),    zeta = r / (B t^beta)
#   with a A^4 = m_pq (flux balance) and a A^3 = B^2 (equation balance),
#   alpha = 1/4, beta = 7/8.

D = physics.diffusivity(p)
k = physics.mobility(p)
m_pq = p.m_q / (2 * np.pi)

rigid = closed_form.exact("neumann", i=1, geometry="radial")
soft = similarity.solve(n=3, bc="neumann", i=1, geometry="radial")

A = (m_pq / k) ** 0.25
B = np.sqrt(k * A**3)

eta = np.linspace(0, 6, 1000)
assert soft.zeta_front is not None  # n = 3, theta_inf = 0 has a front
zeta_front = soft.zeta_front
zeta = np.linspace(soft.zeta[0], zeta_front, 200)  # radial: singular at 0

# %%

CW, GR = 3.25, 1.618
fig = plt.figure(figsize=(2*CW, CW / (GR)), dpi=300, layout="constrained",
                 clear=True, num=1)
ax_w, ax_rigid, ax_soft = fig.subplots(1, 3)

x, w = io.sort_fields(run.x_vert, run.w)
for k_t, color in zip(idx, colors):
    t = run.t[k_t]
    ax_w.plot(x, w[k_t] * M_TO_MICRON, color=color)
    ax_rigid.plot(x / np.sqrt(4*D * t), (w[k_t] - p.w_i) / (m_pq * t / D), color=color)
    ax_soft.plot(x / (B * t**soft.beta), (w[k_t]) / (A * t**soft.alpha), color=color)

ax_rigid.plot(eta, rigid(2*eta), "k--", lw=1.5, label="Rigid")
ax_soft.plot(zeta, soft(zeta), "k--", lw=1.5, label="Soft")

ax_w.axhline(p.w_i * M_TO_MICRON, color="0.6", lw=0.8, ls=":")
ax_w.set(
    xlabel="Radial distance $r$ (m)",
    ylabel=r"Aperture $w$ ($\mu$m)",
    xlim=(0, p.L),
)
ax_rigid.set(
    xlabel=r"$\eta = r/\sqrt{Dt}$",
    ylabel=r"$\phi = D_{0}\,(w - w_0)\,/\,(m_q t / 2\pi)$",
    xlim=(0, 2),
    ylim=(None, 3)
)
ax_soft.set(
    xlabel=r"$\zeta = r/(B\,t^{7/8})$",
    ylabel=r"$\theta = w\,/\,(A\,t^{1/4})$",
    xlim=(0, 2),
    ylim=(None, 3)
)
for ax in (ax_rigid, ax_soft):
    ax.legend(fontsize="small", frameon=False, loc="upper right")
    ax.set_ylim(bottom=0)
ax_w.set_ylim(bottom=0)

annotate_panel_labels([ax_w, ax_rigid, ax_soft], xy=(0.05, 0.98), fontsize="small")

sm = ScalarMappable(norm=norm, cmap=cmap)
fig.colorbar(sm, ax=[ax_w, ax_rigid, ax_soft], label="Time $t$ (days)", pad=0.01)

fig.canvas.draw_idle()
plt.pause(0.01)


fig.savefig("/home/kahmadov/phd/migration/figures/aperture_radial.pdf")
