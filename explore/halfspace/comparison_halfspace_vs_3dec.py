"""Compare front migration from the pure-Python halfspace run against 3DEC.

Pick ``CASE`` to select constant-rate (q_0) or ramp (m_q) injection.
"""

import matplotlib.pyplot as plt

from fracinj import analysis, io, paths, physics, plotting

CASE = "ramp"  # "constant" | "ramp"

CASES = {
    "constant": dict(
        run_3dec=("wi-1e-05", "q-5e-07.hdf5"),
        run_halfspace=("wi-1e-05", "q_0-5e-07.hdf5"),
        subdir=None,
    ),
    "ramp": dict(
        run_3dec=("linear", "run-q-1e-06.hdf5"),
        run_halfspace=("linear", "m_q-1e-06.hdf5"),
        subdir="ramp-injection",
    ),
}
case = CASES[CASE]

# %%
run_3dec = io.read_hdf5(paths.results_dir("3dec", case["run_3dec"][0]) / case["run_3dec"][1])
run_halfspace = io.read_halfspace(
    paths.results_dir("halfspace", case["run_halfspace"][0]) / case["run_halfspace"][1]
)

result_halfspace = analysis.analyze_front(run_halfspace, stress_front=True)
result_3dec = analysis.analyze_front(run_3dec, stress_front=True)

if CASE == "ramp":
    title = rf"$m_q={plotting.sci_latex(run_3dec.params.m_q)}\,\mathrm{{m^2 \cdot s^{{-2}} }}$"
else:
    title = rf"$q_0={plotting.sci_latex(run_3dec.params.q_0)}\,\mathrm{{m^2 \cdot s^{{-1}} }}$"

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
    title=title,
    xscale="log",
    yscale="log",
)
ax.legend(frameon=False)
if case["subdir"]:
    fig.savefig(
        paths.figure_path("comparison-halfspace-3dec", subdir=case["subdir"], ext="png")
    )
fig.canvas.draw_idle()
plt.pause(0.01)

# %%
print(f"Scaling exponent: alpha = {result_halfspace.alpha_emp:.2f}")

if CASE == "constant":
    print(
        "w_0/w_i =",
        run_halfspace.w[len(result_halfspace.t)][0] / run_halfspace.w[0][0],
    )

    print(
        "theta_∞ = (t_c/t)^(1/5) =",
        (physics.critical_time(run_halfspace.params) / result_halfspace.t[-1])
        ** (1 / 5),
    )

    # %%
    w_char, t_char = physics.dimensionalize(run_halfspace.params)
    print("Characteristic time: ", t_char)
    print("Characteristic aperture increase: ", w_char / run_halfspace.params.w_i)
