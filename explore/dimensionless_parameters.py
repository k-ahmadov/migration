import sympy as sp

sp.init_printing(use_unicode=True)

from fracinj import io, paths

# %% constant injection rate

k_n, mu, w_i, q_0, L = sp.symbols("k_n mu w_i q_0 L", positive=True)

p = {k_n: 200e9, w_i: 1e-5, mu: 1e-3, L: 1000, q_0: 1e-4}

a = k_n / (12 * mu)
t_c = a * w_i**5 / q_0**2
t_star = (L**5 / (a * q_0**3)) ** sp.Rational(1, 4)

t_c_val = t_c.subs(p).evalf()
t_star_val = t_star.subs(p).evalf()

print(f"t_c = {t_c_val} s")
print(f"t_star = {t_star_val / 3600 / 24:.2f} day")
print(f"(t_c / t_star)^(1/5) = {(t_c_val / t_star_val) ** sp.Rational(1, 5)}")

# %% ramp injection rate

k_n, mu, w_i, m_q, L = sp.symbols("k_n mu w_i m_q L", positive=True)

p = {k_n: 200e9, w_i: 1e-5, mu: 1e-3, L: 100, m_q: 1e-15}

a = k_n / (12 * mu)
w_char = (L**3 * m_q / a**2) ** sp.Rational(1, 7)
t_char = L**2 / (a * w_char**3)
eps = w_i / w_char

print(f"epsilon = {eps.subs(p).evalf()}")
print(f"t_char = {t_char.subs(p).evalf()}")

# %% validate it with data

filepath = paths.results_dir("3dec", "linear") / "run-q-1e-06.hdf5"
run = io.read_hdf5(filepath)
w_0 = run.w[-1, 0]
t_fin = run.t[-1]

print(f"epsilon = {p[w_i] / w_0}")
print(f"t_fin = {t_fin}")

# %% estimate m_q of soultz

q_beg = 0.15 * 1e-3
q_fin = 36 * 1e-3
t_fin = 16 * 24 * 60 * 60

m_q_field = (q_fin - q_beg) / t_fin
print(f"m_q = {m_q_field} m3.s-2")
