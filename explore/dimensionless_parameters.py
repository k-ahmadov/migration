from pathlib import Path

import sympy as sp

from mypackages import file_io, physics

# %%
result_dir = Path.cwd() / "results" / "3dec" / "wi-1e-05"
run = file_io.read_run(result_dir / "q-5e-07.hdf5")

# %%
p_inj = run.p[-1, 0]
p_char = (12 * run.params.k_n**3 * run.params.L * run.params.q_0 * run.params.mu) ** (
    1 / 4
)
print(p_inj / p_char)

sn_inj = run.sn[-1, 0]
E_ = run.params.E / (1 - run.params.nu**2)
sn_char = (run.params.q_0 / (run.params.L**3 * physics.parameter_a(run.params))) ** (
    1 / 4
) * E_

print(sn_inj / sn_char)

print(sn_inj / p_char)

# %%

k_n, mu, w_i, q_0, L = sp.symbols("k_n mu w_i q_0 L", positive=True)

p = {k_n: 200e9, w_i: 1e-5, mu: 1e-3, L: 1000, q_0: 1e-4}

a = k_n / (12 * mu)
t_c = a * w_i**5 / q_0**2
t_star = (L**5 / (a * q_0**3)) ** sp.Rational(1, 4)

t_c_val = t_c.subs(p).evalf()
t_star_val = t_star.subs(p).evalf()

print(f"t_c = {t_c_val} s")
print(f"t_star = {t_star_val/3600/24:.2f} day")
print(f"(t_c / t_star)^(1/5) = {(t_c_val / t_star_val) ** sp.Rational(1, 5)}")
