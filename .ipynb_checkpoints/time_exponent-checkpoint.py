# %% [markdown]
# # Tensile Stress Peak Migration — Power-Law Scaling Analysis
#
# The tensile stress migrates in time according to a power-law relationship:
#
# \[
# d(t) = c\, t^{\alpha},
# \]
#
# where \(d(t)\) is the peak position, \(c\) is a prefactor, and \(\alpha\) is the time-scaling exponent.
#
# ---
#
# ## Objective
#
# This notebook investigates how the time exponent \(\alpha\) depends on:
#
# - the **applied injection rate** \(q_0\),
# - the **initial fracture aperture** \(w_i\).
#
# ---
#
# ## Base Physical Parameters
#
# | Property | Value |
# | --- | --- |
# | Fracture length, \(L\) | 100 m |
# | Normal stiffness, \(k_n\) | 50 GPa/m |
# | Initial aperture, \(w_i\) | 0.1 mm |
# | Viscosity, \(\mu\) | \(10^{-3}\) Pa·s |
# | Applied injection rate, \(q_0\) | \(10^{-5}\) m\(^2\)/s |
# | Young’s modulus, \(E\) | 60 GPa |
# | Poisson’s ratio, \(\nu\) | 0.25 |
#
# ---
#
# ## Approach
#
# For each simulation:
#
# 1. The tensile stress peak position \(d(t)\) is extracted over time.
# 2. A power law \(d(t) = A t^{\alpha}\) is fitted.
# 3. The exponent \(\alpha\) is analyzed as a function of:
#    - injection rate \(q_0\),
#    - initial aperture \(w_i\) (and corresponding permeability \(k = w_i^2/12\)).
#
# ---
#
# ## Semi-Analytical Comparison
#
# Results are compared to a semi-analytical scaling of the form
#
# \[
# x_f = \zeta_f (a q_0^3 t^4)^{1/5}, 
# \qquad a = \frac{k_n}{12\mu},
# \]
#
# which predicts a theoretical exponent
#
# \[
# \alpha = \frac{4}{5}.
# \]
#
# This comparison evaluates whether the numerically observed tensile-stress migration follows the expected \(t^{4/5}\) scaling.
#
# ---

# %%
from pathlib import Path

import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# %%
# -------------------------------------------------------------------
# Utilities (front + fitting + fiel IO)
# -------------------------------------------------------------------

def power_law(t, A, alpha):
    return A * t**alpha

def find_front(x, sn_tx):
    # front = location of minimum normal stress per time step
    return x[np.argmin(sn_tx, axis=1)]

def cut_before_boundary(front, t, L, buffer=1.0):
    m = front < (L - buffer)
    return front[m], t[m]

def fit_front_power_law(t, front):
    t = np.asarray(t)
    front = np.asarray(front)
    m = t > 0
    t, front = t[m], front[m]
    (A, alpha), _ = curve_fit(power_law, t, front, p0=(front.max(), 0.8), maxfev=20_000)
    return A, alpha

def read_run(fp: Path):
    with h5py.File(fp, "r") as f:
        x_el = f["coordinates"]["x_elastic"][:]
        t = f["coordinates"]["t"][:]
        sn_tx = f["fields"]["stress_normal"][:]
        params = {k: f["parameters"][k][()] for k in f["parameters"].keys()}
    return x_el, t, sn_tx, params

def analyze_run(fp: Path, *, buffer=1.0, t_slice=None):
    x_el, t, sn_tx, params = read_run(fp)
    L = float(params["L"])
    front = find_front(x_el, sn_tx)
    front, t = cut_before_boundary(front, t, L, buffer=buffer)

    if t_slice is not None:
        front = front[t_slice]
        t = t[t_slice]

    A, alpha = fit_front_power_law(t, front)
    return {"file": fp.name, "A": A, "alpha": alpha, "front": front, "t": t, **params}


# %%
# -------------------------------------------------------------------
# Importing results of alpha for injection rate effect analysis
# -------------------------------------------------------------------

result_dir = Path.cwd() / "results" / "fvm-elastic" / "runs"
files_q0 = [
    result_dir / "run-q-1e-06.hdf5",
    result_dir / "run-q-1e-05.hdf5",
    result_dir / "run-q-1e-04.hdf5",
    result_dir / "run-q-1e-03.hdf5",
]

results_q0 = []
for fp in files_q0:
    if not fp.exists():
        print(f"Missing: {fp.name}")
        continue
    results_q0.append(analyze_run(fp, buffer=1.0))

results_q0 = sorted(results_q0, key=lambda d: d["q_0"])
for r in results_q0:
    print(f"{r['file']}: q0={r['q_0']:.1e}  A={r['A']:.4g}  alpha={r['alpha']:.4g}")

# %%
# -------------------------------------------------------------------
# Plot exponent of time alpha versus applied injection rate
# -------------------------------------------------------------------
q0 = np.array([r["q_0"] for r in results_q0])
alpha = np.array([r["alpha"] for r in results_q0])

fig, ax = plt.subplots()
ax.plot(q0, alpha, "x")
ax.set_xscale("log")
ax.set_xlabel(r"Applied injection rate $q_0$, [m$^2$/s]")
ax.set_ylabel(r"Exponent $\alpha$")
ax.grid(True, which="both", alpha=0.2)
plt.show()


# %%
# -------------------------------------------------------------------
# Importing results of alpha for initial aperture analysis
# -------------------------------------------------------------------
files_wi = [
    result_dir / "run-wi-1e-05.hdf5",
    result_dir / "run-wi-5e-05.hdf5",
    result_dir / "run-q-1e-05.hdf5",  # baseline
]

results_wi = []
for fp in files_wi:
    if not fp.exists():
        print(f"Missing: {fp.name}")
        continue
    r = analyze_run(fp, buffer=1.0)
    if "w_i" not in r:
        print(f"No w_i in {fp.name}")
        continue
    results_wi.append(r)

results_wi = sorted(results_wi, key=lambda d: d["w_i"])
for r in results_wi:
    print(f"{r['file']}: w_i={r['w_i']:.1e}  A={r['A']:.4g}  alpha={r['alpha']:.4g}")

# %%
# -------------------------------------------------------------------
# Plot exponent of time alpha versus initial aperture
# -------------------------------------------------------------------
wi = np.array([r["w_i"] for r in results_wi])
alpha = np.array([r["alpha"] for r in results_wi])

fig, ax = plt.subplots()
ax.plot(wi, alpha, "x")
ax.set_xscale("log")
ax.set_xlabel(r"Initial aperture $w_i$, [m]")
ax.set_ylabel(r"Exponent $\alpha$")
ax.grid(True, which="both", alpha=0.2)

secax = ax.secondary_xaxis(
    "top",
    functions=(lambda w: w**2/12.0, lambda k: np.sqrt(12.0*k)),
)
secax.set_xlabel(r"Permeability $k=w_i^2/12$ [m$^2$]")
secax.set_xscale("log")

plt.show()

# %% [markdown]
# # Semi-analytical solution for aperture front
# $$x_f = \zeta_f (a q^3 t^4)^{1/5}$$, where $a = k_n / (12 \mu)$.
# where $\zeta_f \approx 1$

# %%
# -------------------------------------------------------------------
# Comparison of numerical results with Murphy style semi-analytical solution
# -------------------------------------------------------------------

r = analyze_run(result_dir / "run-wi-1e-05.hdf5", buffer=1.0)

alpha_semi = 4/5
a = r["k_n"] / (12.0 * r["mu"])
A_semi = (a * r["q_0"]**3)**(1/5)

print(f"fit:  A={r['A']:.4g}, alpha={r['alpha']:.4g}")
print(f"semi: A={A_semi:.4g}, alpha={alpha_semi:.4g}")

fig, ax = plt.subplots()
ax.plot(r["t"], r["front"], ".", label="numerical front")
ax.plot(r["t"], power_law(r["t"], r["A"], r["alpha"]), label="power-law fit")
ax.plot(r["t"], power_law(r["t"], A_semi, alpha_semi), label="semi-analytical")
ax.set_xlabel("t [s]")
ax.set_ylabel("tensile peak position [m]")
ax.grid(True, alpha=0.1)
ax.legend()
plt.show()
