import sympy as sp

# Symbols
x, D = sp.symbols('x D', positive=True)
C1, C2 = sp.symbols('C1 C2')

# Proposed solution
f = sp.exp(-x**2/(4*D)) * (
    C1 + C2*sp.erfi(x/(2*sp.sqrt(D)))
)

print(f)

# Compute the ODE
ode = sp.simplify(
    D*sp.diff(f, x, 2)
    + x*sp.diff(f, x)/2
    + f/2
)

f = sp.Function('f')
eq = sp.Eq(
    D*sp.diff(f(x), x, 2)
    + x*sp.diff(f(x), x)/2
    + f(x)/2,
    0
)

print(ode)
print(sp.dsolve(eq))
