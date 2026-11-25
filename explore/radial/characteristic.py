import numpy as np

from fracinj.types import Parameters

# %%

p = Parameters(
    k_n=200e9,
    L=3000.0,
    mu=1e-3,
    w_i=1e-5,
    E=60e9,
    nu=0.25,
    Nx_p=2048,
    Nx_sn=2048,
    Nt=10000,
    m_q=1e-10,
    radial=1.0,
)

a = p.k_n / (12 * p.mu)
w_star = ((p.m_q * p.L**2) / (2 * np.pi * a**2)) ** (1 / 7)
t_star = p.L**2 / (a * w_star**3)

D0 = a * p.w_i**3

tc = 2 * np.pi * D0 * p.w_i / p.m_q


print(f"characteristic time for length L, {t_star:.2e}", )
print(f"characteristic time for initial aperture w0, {tc:.2e}")
print("ratio of characteristic deformation for length L,", (t_star / tc)**(1/4))
