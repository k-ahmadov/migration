from pathlib import Path

import matplotlib.pyplot as plt

from mypackages import file_io, front_analysis, physics, types

# %%
result_dir = Path.cwd() / "results" / "fvm-elastic" / "wi-1e-05"
filename = "run-q-5e-07.hdf5"
filepath = result_dir / filename

run = file_io.read_fvm(filepath=filepath)

result = front_analysis.analyze(run, stress_front=True, slc=slice(None, None))
result_rigid = front_analysis.analyze_rigid(run, stress_front=True)

result_soft = front_analysis.analyze_soft(run, stress_front=True)

# %%
fig, ax = plt.subplots(figsize=(6.4 / 1.4, 4.8 / 1.4), dpi=200)
ax.loglog(result.t_front, result.x_front, ".")
ax.loglog(result_soft.t_front, result_soft.x_analytical(), "-", label="soft")
ax.loglog(result_rigid.t_front, result_rigid.x_analytical(), "-", label="rigid")
ax.set(xlabel="Time [s]", ylabel="Distance [m]", title="Tensile peak migration")
ax.legend()
fig.tight_layout()
plt.show()

# %%

print(f"Scaling exponent: α = {result.α_emp:.2f}")

print("w_0/w_i =", run.w[len(result.t_front)][0] / run.w[0][0])

print(
    "θ_∞ = (t_c/t)^(1/5) =",
    (physics.characteristic_time(run.params) / result.t_front[-1]) ** (1 / 5),
)

# %%
# params = types.Parameters(
#     E=60e9, L=100, k_n=200e9, mu=1e-3, nu=0.25, w_i=1e-5, q_0=1e-8
# )
w_char, t_char = physics.dimensionalize(run.params)
print("Characteristic time: ", t_char)
print("Characteristic aperture increase: ", w_char / run.params.w_i)
