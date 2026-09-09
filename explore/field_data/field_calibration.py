"""Back out (k_n, q_0, epsilon) from the fitted field-migration prefactors.

Moved out of scripts/fig5_field_migration.py -- this is exploratory
calibration, not part of the figure.
"""

# %%
import sympy as sp

from fracinj.math_utils import fit_power_law_fixed_exponent
from fracinj.analysis import RIGID, SOFT
from scripts.fig5_field_migration import (
    d_pct_s,
    percentile_envelope,  # noqa: F401  (kept for interactive re-use)
    t_pct_s,
)


def prefactors(t_pct, d_pct, split_frac):
    idx = int(len(t_pct) * split_frac)
    A_e = fit_power_law_fixed_exponent(t_pct[:idx], d_pct[:idx], RIGID.front_exponent)
    A_l = fit_power_law_fixed_exponent(t_pct[idx:], d_pct[idx:], SOFT.front_exponent)
    return A_e, A_l


# %%
prefactor_early, prefactor_late = prefactors(t_pct_s, d_pct_s, split_frac=1 / 4)

D = prefactor_early**2 / 4
mu = 1e-3
w_i = 1e-4
k_n = 12 * D * mu / w_i**3
print(f"k_n = {k_n / 1e9:.5f} GPa/m")

a = k_n / 12 / mu
q_0 = (prefactor_late**5 / a) ** (1 / 3)
print(f"q_0 = {q_0:.1e} m^2/s")

L = 1e3
epsilon = w_i / (L * q_0 / a) ** (1 / 4)
print(f"epsilon = {epsilon}")

# %%
A_late, k_n_s, mu_s, q_0_s = sp.symbols("A_late k_n mu q_0", positive=True)
A_late_eq = sp.Eq(A_late, (k_n_s / (12 * mu_s)) ** (1 / 5) * q_0_s ** (3 / 5))
k_n_expr = sp.solve(A_late_eq, k_n_s)[0]
print(k_n_expr.subs({A_late: prefactor_late, mu_s: 1e-3, q_0_s: 1e-4}))
