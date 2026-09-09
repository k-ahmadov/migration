import numpy as np
import matplotlib.pyplot as plt

from mysolvers import aperture_solver, exact_solutions

# %%

num_nodes = 1000
num_steps = 1000
w_i = 1e-5
w_b = 2e-5
wi_hat = w_i / w_b
r_b = 0.1
L = 100
rb_hat = r_b / L
t_hat_final = 0.1
w_hat_tx = aperture_solver.solve_linear_radial_diffusion(
    num_nodes=num_nodes,
    num_steps=num_steps,
    w_initial=wi_hat,
    r_b=rb_hat,
    t_final=t_hat_final,
)


# %%
w_tx = w_hat_tx * w_b
D = 1
t = np.linspace(0, t_hat_final * L**2 / D, num_steps)
r = np.linspace(r_b, L, num_nodes)


# %%

zeta_b = r_b / 100
zeta = np.linspace(zeta_b, 5, 400)
theta = exact_solutions.solve_linear_radial_diffusion(
    theta_inf=wi_hat, zeta_b=1e-3, zeta=zeta
)

# %%

fig = plt.figure(figsize=(6.4/1.5, 4.8/1.5), dpi=150, layout='tight', clear=True, num=1)
ax = fig.subplots()
for i in range(10, num_steps, 20):
    zeta_num = r / np.sqrt(4 * D * t[i])
    theta_num = w_tx[i] / w_b
    ax.plot(zeta_num, theta_num, ':')
ax.plot(zeta, theta, 'k-')
ax.set(
    xlabel = r"$\zeta$",
    ylabel = r"$\theta$",
    title = "Dimensionless aperture profile",
    xlim=(-0.1, 1)
)
fig.canvas.draw_idle()
plt.pause(0.01)




