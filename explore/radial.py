import matplotlib.pyplot as plt
import numpy as np

from fracinj import physics
from fracinj.solvers import aperture, exact
from fracinj.types import Parameters

# %%

params = Parameters(
    k_n=200e9,
    L=100.0,
    mu=1e-3,
    w_i=1e-5,
    E=60e9,
    nu=0.25,
    Nx_p=1024,
    Nx_sn=1024,
    Nt=10000,
    radial=1.0,
    q_0=1e-8,
)

scales = physics.cubic_scales(params, pin="length")
nx, nt = int(params.Nx_p), int(params.Nt)
W_i = params.w_i / scales.w_c
t_star = params.L**2 / (physics.mobility(params) * scales.w_c)


# %%

w = aperture.solve_radial_diffusion(
    num_nodes=nx,
    num_steps=nt,
    w_initial=W_i,
    t_final=t_star,
    k_func=lambda w: w**3,
    left_bc_constant_rate=physics.is_constant_rate(params),
    t_stop=params.t_stop / scales.t_c if params.t_stop > 0 else None,
)

# Linear (Theis) solution, linearised about w_i: k(w) = w^3 -> D_hat = W_i^3, and
# the source -R W^3 W_R = 1 -> -R D_hat W_R = 1. In the Theis variables
#   zeta = R / (2 sqrt(D_hat T)),   phi = 2 D_hat (W - W_i)
# the solution is phi = E1(zeta^2). Valid where W - W_i << W_i (away from the source).
D_hat = W_i**3
zeta = np.linspace(1e-3, 3, 400)
phi = exact.solve_linear_radial_diffusion_const_flux(zeta)

# %%

n_profiles = 5
idx = np.linspace(0, len(w) - 1, n_profiles + 1).astype(int)[1:]
r = np.linspace(0, 1, nx)
# solver rows are T_hat = dt, 2 dt, ..., T_hat_final (initial condition dropped)
t = np.linspace(t_star / (nt - 1), t_star, nt - 1)

# %%

CW, GR = 3.25, 1.618
fig = plt.figure(
    figsize=(2 * CW, CW / GR), dpi=200, layout="constrained", clear=True, num=1
)
ax1, ax2 = fig.subplots(1, 2)
for i in idx:
    ax1.plot(r, w[i] - W_i)
    ax2.plot(r / (2 * np.sqrt(D_hat * t[i])), 2 * D_hat * (w[i] - W_i))
ax2.plot(zeta, phi, "k--", lw=1, label=r"linear (Theis), $D = a w_i^3$")
ax2.legend(fontsize=6, frameon=False)
ax1.set(
    xlabel=r"$\hat{r}$",
    ylabel=r"$\hat{w} - \hat{w}_i$",
    xlim=(0, None),
    ylim=(0, None)
)
ax2.set(
    xlabel=r"$\zeta = \hat{r} / 2\sqrt{\hat{D}\hat{t}}$",
    ylabel=r"$2\hat{D}(\hat{w} - \hat{w}_i)$",
    xlim=(0, zeta[-1]),
    ylim=(0, 1.2 * 2 * D_hat * (w[idx] - W_i).max()),  # E1 is log-singular at the source
)
fig.canvas.draw_idle()
plt.pause(0.01)

# %%
