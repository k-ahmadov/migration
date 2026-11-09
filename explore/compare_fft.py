import numpy as np

from fracinj.solvers.elastic import _product_trapz_weights_cached

# --- problem data (SI units) ---
E_p, kn = 30e9, 10e9  # E' [Pa], k_n [Pa/m]
p0, b = 1e6, 0.2  # pressure increment [Pa], width scale [m]
lam = 4 * kn / (np.pi * E_p)  # [1/m], same lam as in your solver
ell = 1 / (np.pi * lam)  # = E'/(4 k_n) [m]
Q = p0 * b * np.sqrt(np.pi)  # = integral of p


def p_func(x):
    return p0 * np.exp(-((x / b) ** 2))


def solve_truncated(L, n):
    """Your scheme on the physical interval [-L, L]."""
    xt = np.linspace(-1, 1, n + 1)
    Kw = _product_trapz_weights_cached(n)
    w = np.full(n + 1, 2 / n)
    w[[0, -1]] /= 2  # trapezoid weights on [-1, 1]
    M = np.eye(n + 1) - lam * L * (Kw + np.log(L) * w[None, :])
    x = L * xt
    sigma = np.linalg.solve(M, p_func(x))
    return x, sigma, (p_func(x) - sigma) / kn  # v from the joint law


def solve_fourier(L_fft, N=2**20):
    """Exact infinite-fracture solution, inverted by FFT."""
    x = (np.arange(N) - N // 2) * (L_fft / N)
    xi = 2 * np.pi * np.fft.fftfreq(N, d=L_fft / N)
    p_hat = np.fft.fft(np.fft.ifftshift(p_func(x)))
    a = ell * np.abs(xi)
    sigma = np.fft.fftshift(np.fft.ifft(p_hat * a / (1 + a))).real
    v = np.fft.fftshift(np.fft.ifft(p_hat / (kn * (1 + a)))).real
    return x, sigma, v


xf, sf, vf = solve_fourier(L_fft=4000 * max(b, ell))

win = 0.5  # compare on a common window |x| <= 0.5
for L in [0.5, 1, 2, 4, 8]:
    n = int(50 * L / b)  # keep the resolution fixed as L grows
    x, s, v = solve_truncated(L, n)
    m = np.abs(x) <= min(win, L)
    es = np.max(np.abs(s[m] - np.interp(x[m], xf, sf))) / p0
    ev = np.max(np.abs(v[m] - np.interp(x[m], xf, vf))) / (p0 / kn)
    P = np.trapezoid(s, x) / Q  # np.trapz on NumPy < 2.0
    print(f"L={L:4}  n={n:5}  sigma err={es:.2e}  v err={ev:.2e}  P_trunc/Q={P:+.3f}")
