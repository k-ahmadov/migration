import matplotlib.pyplot as plt

from fracinj import analysis, io, paths, physics, plotting

# %%
run_3dec = io.read_hdf5(paths.results_dir("3dec", "linear") / "run-q-1e-06.hdf5")
run_halfspace = io.read_halfspace(
    paths.results_dir("halfspace", "linear") / "m_q-1e-06.hdf5"
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
    title=rf"$m_q={plotting.sci_latex(run_3dec.params.m_q)}\,\mathrm{{m^2 \cdot s^{{-2}} }}$",
    xscale="log",
    yscale="log",
)
ax.legend(frameon=False)
fig.canvas.draw_idle()
plt.pause(0.01)


# %%

print(f"Scaling exponent: alpha = {result_halfspace.alpha_emp:.2f}")
