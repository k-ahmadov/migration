# %%
"""
Early-time (T << 1) similarity solution of the linearized aperture equation.

Dimensionless problem (hats dropped):
    w_t = (w^3 w_x)_x,   -w^3 w_x|_{x=0} = t,   w -> 1 (x -> oo),   w(x, 0) = 1.

For t << 1, w - 1 << 1 and w^3 ~ 1, so the deviation p = w - 1
(dimensionless pressure) obeys the linear problem
    p_t = p_xx,   -p_x|_{x=0} = t,   p -> 0 (x -> oo),   p(x, 0) = 0.

Only the deviation is self-similar:  p = t^(3/2) f(eta),  eta = x / (2 sqrt t),
so that  w = 1 + t^(3/2) f(eta)  with  f = 8 i^3erfc(eta).
"""

import sympy as sp

sp.init_session()

# %% ---------------------------------------------------------------
# 0) Symbols and linearized PDE  p_t = p_xx  for the deviation p = w - 1
# ---------------------------------------------------------------
x = sp.symbols("x", real=True)
t = sp.symbols("t", positive=True)
eta = sp.symbols("eta", real=True)

p, f, g = sp.Function("p"), sp.Function("f"), sp.Function("g")

pde = sp.Eq(sp.Derivative(p(x, t), t), sp.Derivative(p(x, t), x, 2))

# %% ---------------------------------------------------------------
# 1) Similarity ansatz  p = w - 1 = t^(3/2) f(eta),  eta = x / (2 sqrt t)
# ---------------------------------------------------------------
ansatz = t**sp.Rational(3, 2) * f(x / (2 * sp.sqrt(t)))

expr = (pde.lhs - pde.rhs).subs(p(x, t), ansatz).doit()
expr = expr.subs(x, 2 * eta * sp.sqrt(t)).doit()  # x -> eta, Subs -> f'(eta)
expr = sp.simplify(expr / sp.sqrt(t))             # remove common t^(1/2)

ode = sp.Eq(sp.expand(-4 * expr), 0)
print("Reduced ODE:")
sp.pprint(ode)                                    # f'' + 2 eta f' - 6 f = 0

# %% ---------------------------------------------------------------
# 2) Differentiate 3 times (the f''' terms cancel), then set g = f'''
# ---------------------------------------------------------------
d3 = sp.diff(ode.lhs, eta, 3)                     # f^(5) + 2 eta f^(4)

ode_g = sp.Eq(
    d3.subs(
        {
            sp.Derivative(f(eta), (eta, 5)): sp.Derivative(g(eta), (eta, 2)),
            sp.Derivative(f(eta), (eta, 4)): sp.Derivative(g(eta), eta),
        },
        simultaneous=True,
    ),
    0,
)
sol_g = sp.dsolve(ode_g, g(eta))                  # g = C1 + C2 erf(eta)
print("\nODE for g = f''':")
sp.pprint(ode_g)
sp.pprint(sol_g)

# %% ---------------------------------------------------------------
# 3) Integrate 3 times WITH integration constants
# ---------------------------------------------------------------
C1, C2, C3, C4, C5 = sp.symbols("C1:6")

F = (sp.integrate(sol_g.rhs, eta, eta, eta)
     + C3 * eta**2 / 2 + C4 * eta + C5)

# %% ---------------------------------------------------------------
# 4) Put F back into the original ODE to fix C3, C4, C5
#    (the residual is a polynomial of degree <= 2 in eta)
# ---------------------------------------------------------------
res = ode.lhs.subs(f(eta), F).doit()
res = sp.expand(sp.simplify(res))
conds = [sp.diff(res, eta, k).subs(eta, 0) for k in range(3)]
F = F.subs(sp.solve(conds, [C3, C4, C5], dict=True)[0])
assert sp.simplify(ode.lhs.subs(f(eta), F).doit()) == 0

# %% ---------------------------------------------------------------
# 5) Far-field (and initial) condition: p -> 0  =>  f -> 0 as eta -> oo.
#    The eta^3 growth must cancel: C1 = -C2
# ---------------------------------------------------------------
F = sp.simplify(F.subs(C1, -C2))
assert sp.limit(F, eta, sp.oo) == 0

# %% ---------------------------------------------------------------
# 6) Injection condition: -p_x|_{x=0} = t
#    p_x = t^(3/2) f'(eta) / (2 sqrt t) = t f'(eta) / 2  =>  f'(0) = -2
# ---------------------------------------------------------------
C2val = sp.solve(sp.Eq(sp.diff(F, eta).subs(eta, 0), -2), C2)[0]
F = sp.simplify(F.subs(C2, C2val))
print("\nSolution f(eta):")
sp.pprint(F)
print("\nf(0) =", sp.simplify(F.subs(eta, 0)))     # 4 / (3 sqrt(pi))

# %% ---------------------------------------------------------------
# 7) Check against the closed form  f = 8 i^3erfc(eta)
# ---------------------------------------------------------------
i3erfc = ((eta**2 + 1) * sp.exp(-eta**2) / sp.sqrt(sp.pi)
          - eta * (2 * eta**2 + 3) * sp.erfc(eta) / 2) / 6
assert all(abs(sp.N((F - 8 * i3erfc).subs(eta, v))) < 1e-12
           for v in [0, 0.5, 1, 2, 4])
print("Matches 8 i^3erfc(eta).")

# %% ---------------------------------------------------------------
# 8) Back to the aperture:  w = 1 + t^(3/2) f(eta)
# ---------------------------------------------------------------
w_early = 1 + t**sp.Rational(3, 2) * F.subs(eta, x / (2 * sp.sqrt(t)))
w0_inj = sp.simplify(w_early.subs(x, 0))
print("\nAperture at the injection point: w(0, t) =", w0_inj)

# Sanity checks on the full solution
assert sp.simplify(sp.diff(w_early, t) - sp.diff(w_early, x, 2)) == 0      # PDE
assert sp.simplify(-sp.diff(w_early, x).subs(x, 0) - t) == 0              # flux
print("PDE and injection condition verified for w = 1 + t^(3/2) f(eta).")

# %%
