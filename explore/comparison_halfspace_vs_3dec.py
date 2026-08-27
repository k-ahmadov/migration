from pathlib import Path

import matplotlib.pyplot as plt

from mypackages import file_io, front_analysis, physics, plotting

# %%
result_halfspace_dir = Path.cwd() / "results" / "halfspace" / "wi-1e-05"
result_3dec_dir = Path.cwd() / "results" / "3dec" / "wi-1e-05"

run_halfspace = file_io.read_halfspace(result_halfspace_dir / "q-5e-07.hdf5")
run_3dec = file_io.read_run(result_3dec_dir / "q-5e-07.hdf5")

result_halfspace = front_analysis.analyze(
    run_halfspace, stress_front=True, slc=slice(None, None)
)
result_3dec = front_analysis.analyze(run_3dec, stress_front=True, slc=slice(None, None))

# %%
fig = plt.figure(
    figsize=(6.4 / 1.5, 4.8 / 1.5), dpi=150, layout="tight", clear=True, num=1
)
ax = fig.subplots()
ax.plot(result_3dec.t_front, result_3dec.x_front, ".", label="3DEC")
ax.plot(result_halfspace.t_front, result_halfspace.x_front, "1", label="Halfspace")
ax.set(
    xlabel="Time [s]",
    ylabel="Front position [m]",
    title=rf"$q_0={plotting.sci_latex(run_3dec.params.q_0)}\,\mathrm{{m^2 \cdot s^{{-1}} }}$",
    xscale="log",
    yscale="log",
)
ax.legend(frameon=False)
fig.canvas.draw_idle()
plt.pause(0.01)


# %%

print(f"Scaling exponent: alpha = {result_halfspace.alpha_emp:.2f}")

print(
    "w_0/w_i =",
    run_halfspace.w[len(result_halfspace.t_front)][0] / run_halfspace.w[0][0],
)

print(
    "theta_∞ = (t_c/t)^(1/5) =",
    (physics.critical_time(run_halfspace.params) / result_halfspace.t_front[-1])
    ** (1 / 5),
)

# %%
# params = types.Parameters(
#     E=60e9, L=100, k_n=200e9, mu=1e-3, nu=0.25, w_i=1e-5, q_0=1e-8
# )
w_char, t_char = physics.dimensionalize(run_halfspace.params)
print("Characteristic time: ", t_char)
print("Characteristic aperture increase: ", w_char / run_halfspace.params.w_i)
