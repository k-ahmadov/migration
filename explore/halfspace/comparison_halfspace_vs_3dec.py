import matplotlib.pyplot as plt

from fracinj import analysis, io, paths, physics, plotting

# %%
run_3dec = io.read_hdf5(paths.results_dir("3dec", "wi-1e-05") / "q-5e-07.hdf5")
run_halfspace = io.read_halfspace(
    paths.results_dir("halfspace", "wi-1e-05") / "q_0-5e-07.hdf5"
)

result_halfspace = analysis.analyze_front(run_halfspace, stress_front=True)
result_3dec = analysis.analyze_front(run_3dec, stress_front=True)

# %%
fig = plt.figure(
    figsize=(6.4 / 1.5, 4.8 / 1.5), dpi=150, layout="tight", clear=True, num=1
)
ax = fig.subplots()
ax.plot(result_3dec.t, result_3dec.y, ".", label="3DEC")
ax.plot(result_halfspace.t, result_halfspace.y, "1", label="Halfspace")
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
    run_halfspace.w[len(result_halfspace.t)][0] / run_halfspace.w[0][0],
)

print(
    "theta_∞ = (t_c/t)^(1/5) =",
    (physics.critical_time(run_halfspace.params) / result_halfspace.t[-1]) ** (1 / 5),
)

# %%
# params = types.Parameters(
#     E=60e9, L=100, k_n=200e9, mu=1e-3, nu=0.25, w_i=1e-5, q_0=1e-8
# )
w_char, t_char = physics.dimensionalize(run_halfspace.params)
print("Characteristic time: ", t_char)
print("Characteristic aperture increase: ", w_char / run_halfspace.params.w_i)
