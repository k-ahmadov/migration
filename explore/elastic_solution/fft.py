import matplotlib.pyplot as plt

from fracinj import physics
from fracinj.solvers.elastic_fft import solve_elastic_joint
from fracinj.solvers.exact import solve_nonlinear_diffusion_ramp_flux
from fracinj.types import Parameters

# %%
zf, zeta, Pi = solve_nonlinear_diffusion_ramp_flux(extend=5)

params = Parameters(E=60e9, k_n=100e9, nu=0.25)
E_plane = physics.plane_strain_modulus(params)
eps = E_plane / (4 * params.k_n * zf)

Sigma, _ = solve_elastic_joint(zeta, Pi, eps=eps)


# %%

CW, GR = 3.4, 1.618
fig = plt.figure(
    figsize=(CW, CW / GR), dpi=150, layout="constrained", clear=True, num=1
)
ax = fig.subplots()
ax.plot(zeta, Pi)
ax.plot(zeta, Sigma)
ax.set(
    xlabel=r"$x$",
    ylabel="y label",
)
fig.canvas.draw_idle()
plt.pause(0.01)
