"""
Normal stress change on a fluid-pressurized fracture: an elastic joint (normal
stiffness k_n) between two elastic half-spaces (modulus E' = E/(1-nu^2)),
normal loading only, no shear traction.

    Terzaghi + linear joint law:   dsigma_n - dp = -k_n v          (v = opening)

Three solvers, all for the same physics:

1. ``solve_elastic_joint``          plane strain, Fourier transform (FFT)
2. ``solve_elastic_joint_axisym``   axisymmetric, quasi-discrete Hankel transform
3. ``FIE_log_sing``                 plane strain, product-integration solve of
                                    the log-kernel Fredholm equation (legacy,
                                    bounded domain [-1, 1])

NONDIMENSIONALIZATION (shared by 1 and 2)
-----------------------------------------
    X = x/L_p (or R = r/L_p),  Pi = dp/P,  Sigma = dsigma_n/P,  V = k_n v/P
    eps = L_k/L_p = E' / (4 k_n L_p)    (L_k = E'/(4 k_n): elasto-joint length,
                                         L_p = length scale of the pressure pulse)

In transform space both geometries reduce to the same exact multipliers:

    Sigma~ = eps k / (1 + eps k) * Pi~,     V~ = Pi~ / (1 + eps k)

with k = |xi| (Fourier wavenumber) in plane strain and the Hankel wavenumber in
the axisymmetric case. For eps << 1, Sigma ~ eps * Hilbert[dPi/dX] and V ~ Pi.

Both transform solvers are linear in Pi and accept Pi of shape (n,) or
(n_t, n): the last axis is space, so a whole time history is solved at once.
"""

import warnings
from functools import lru_cache

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.special import j0, j1, jn_zeros

DECAY_TOL = 1e-3  # |Pi| at the domain end, relative to max|Pi|, above which we warn


def _warn_if_not_decayed(Pi_end: np.ndarray, Pi: np.ndarray, where: str) -> None:
    peak = np.max(np.abs(Pi), axis=-1)
    if np.any(np.abs(Pi_end) > DECAY_TOL * peak):
        warnings.warn(f"Pi has not decayed at {where}; extend the domain", stacklevel=3)


# ============================================================
# 1. Plane strain: FFT
# ============================================================
def solve_elastic_joint(X: np.ndarray, Pi: np.ndarray, eps: float, pad=4):
    """
    Plane strain. Elasticity (both walls):

        v(x) = -4/(pi E') Int dsigma_n(s) ln|x-s| ds

    so that  eps*Sigma(X) - (1/pi) Int Sigma(S) ln|X-S| dS = eps*Pi(X),  V = Pi - Sigma.
    The net load is zero: Int Sigma dX = 0.

    X   : uniform grid in units of L_p (increasing)
    Pi  : dimensionless pressure change on that grid, shape (n,) or (n_t, n);
          must decay to ~0 at both ends (an isolated pulse)
    eps : L_k / L_p = E' / (4 k_n L_p)
    pad : the FFT box is at least `pad` times the length of the data

    Returns Sigma, V with the shape of Pi.
    """
    X = np.asarray(X, dtype=float)
    Pi = np.asarray(Pi, dtype=float)
    if X.ndim != 1 or Pi.shape[-1] != X.size:
        raise ValueError("X must be 1-D and match the last axis of Pi")
    if np.any(np.diff(X) <= 0):
        raise ValueError("X must be strictly increasing")
    _warn_if_not_decayed(np.maximum(abs(Pi[..., 0]), abs(Pi[..., -1])), Pi, "the domain ends")

    n = X.size           # number of grid points
    h = X[1] - X[0]      # grid spacing

    # Zero-padded FFT length: long enough for the data AND for the kernel width
    # (about eps), rounded up to a power of 2 for speed
    n_needed = max(pad * n, 50 * eps / h)
    nfft = int(2 ** np.ceil(np.log2(n_needed)))

    # Angular wavenumbers matching the FFT output
    k = 2 * np.pi * np.fft.rfftfreq(nfft, d=h)

    # Transform Pi (numpy appends the zeros up to nfft)
    Pi_hat = np.fft.rfft(Pi, nfft, axis=-1)

    # Multiply by the exact solution operators, then transform back
    Sigma_hat = eps * k / (1 + eps * k) * Pi_hat
    V_hat = Pi_hat / (1 + eps * k)

    # Keep only the first n points, which is the original (unpadded) domain
    Sigma = np.fft.irfft(Sigma_hat, nfft, axis=-1)[..., :n]
    V = np.fft.irfft(V_hat, nfft, axis=-1)[..., :n]
    return Sigma, V


# ============================================================
# 2. Axisymmetric: quasi-discrete Hankel transform
# ============================================================
def solve_elastic_joint_axisym(r, Pi, eps, r_eval=None, pad=4, n_max=8000, chunk=512):
    """
    Axisymmetric (radially symmetric pressure change). Elasticity (two
    half-spaces, Murdoch & Germanovich 2012, eq. 11):

        v(r) = 8/(pi E') Int_0^inf dsigma_n(lam) * lam/(lam+r) * K(2 sqrt(lam r)/(lam+r)) dlam

    Zero-order Hankel transform f~(k) = Int_0^inf f(R) J0(kR) R dR turns this
    into the plane-strain multipliers with |xi| -> k.

    Numerics: quasi-discrete Hankel transform (Guizar-Sicairos & Gutierrez-Vega
    2004). Pi is sampled at the Bessel-zero radii R_n = j_n R_box / j_{N+1},
    transformed by a matrix product, multiplied, and summed back as a
    Fourier-Bessel series, which can be evaluated at any radius (also beyond
    the pressurized zone). The whole chain is one linear operator, built once
    and applied to every row of Pi.

    r      : increasing radii from (or near) 0 to r_max, in units of L_p
    Pi     : dimensionless pressure on r, shape (n,) or (n_t, n) (must decay
             to ~0 at r_max) -- or a function Pi(R), which is then evaluated exactly
    eps    : L_k / L_p = E' / (4 k_n L_p)
    r_eval : radii where Sigma, V are returned (default: r); may go beyond r_max
    pad    : Bessel box radius R_box = max(pad * r_max, 50 * eps)
    n_max  : cap on the number of modes (memory ~ 8 n_max^2 bytes)
    chunk  : rows of Pi interpolated at a time (bounds memory for long histories)

    Returns Sigma, V of shape (len(r_eval),) or (n_t, len(r_eval)).
    """
    r = np.asarray(r, float)
    Pi_func = Pi if callable(Pi) else None
    Pi_r = np.asarray(Pi_func(r) if Pi_func is not None else Pi, float)
    if Pi_r.shape[-1] != r.size:
        raise ValueError("the last axis of Pi must match r")
    _warn_if_not_decayed(Pi_r[..., -1], Pi_r, "r_max")

    h = np.min(np.diff(r))                      # resolution to keep
    R_box = max(pad * r[-1], 50 * eps)          # Pi = 0 for R > r_max (zero padding)
    N = int(R_box / h)                          # number of Bessel modes
    if N > n_max:
        N, R_box = n_max, n_max * h
        warnings.warn(
            f"box capped at R = {R_box:.3g} (< 50 eps); V far from the pulse loses "
            "accuracy, Sigma near it is fine. Coarsen r or raise n_max.",
            stacklevel=2,
        )
    Sig_op, V_op, R_n = _hankel_operators(N, R_box, eps, r if r_eval is None else r_eval)

    # Pi at the Bessel-zero radii; zero outside r_max
    if Pi_func is not None:
        Pi_n = np.nan_to_num(np.asarray(Pi_func(R_n), float))
        return Pi_n @ Sig_op, Pi_n @ V_op

    rows = Pi_r.reshape(-1, r.size)
    Sigma = np.empty((rows.shape[0], Sig_op.shape[1]))
    V = np.empty_like(Sigma)
    for i in range(0, rows.shape[0], chunk):
        block = rows[i : i + chunk]
        interp = PchipInterpolator(r, block, axis=-1, extrapolate=False)
        Pi_n = np.nan_to_num(interp(R_n))
        Sigma[i : i + chunk] = Pi_n @ Sig_op
        V[i : i + chunk] = Pi_n @ V_op
    out_shape = Pi_r.shape[:-1] + (Sig_op.shape[1],)
    return Sigma.reshape(out_shape), V.reshape(out_shape)


def _hankel_operators(N, R_box, eps, r_eval):
    """Matrices mapping Pi at the sampling radii R_n to Sigma, V at r_eval.

    Forward transform  Pi~_m = (2/K^2) sum_n J0(j_m j_n / S) Pi_n / J1(j_n)^2,
    multiplier, then the series  f(r) = sum_m f~_m J0(k_m r) 2 / (R_box^2 J1(j_m)^2).
    """
    j = jn_zeros(0, N + 1)
    S, jN = j[-1], j[:-1]
    R_n = jN * R_box / S                        # sampling radii
    k = jN / R_box                              # wavenumbers (1/L_p)
    K = S / R_box                               # max wavenumber
    J1sq = j1(jN) ** 2

    r_eval = np.asarray(r_eval, float)
    # series rows, scaled by the multipliers: (N, n_eval)
    B_T = j0(np.outer(k, r_eval)) * (2 / (R_box**2 * J1sq))[:, None]
    T = j0(np.outer(jN, jN) / S)                # symmetric forward kernel
    pre = (2 / K**2) / J1sq                     # forward weights on Pi_n
    Sig_op = pre[:, None] * (T @ ((eps * k / (1 + eps * k))[:, None] * B_T))
    V_op = pre[:, None] * (T @ ((1 / (1 + eps * k))[:, None] * B_T))
    return Sig_op, V_op, R_n


# ============================================================
# 3. Plane strain, bounded domain: Fredholm integral equation (legacy)
# ============================================================
def FIE_log_sing(lam, RHS, n=2**9, check_cond=True):
    """
    Solve a general Fredholm Integral Equation (FIE) of the second kind
    with a logarithmic kernel using the product integration (Atkinson) method:

        f(x) = g(x) + lam * int_{-1}^{1} ln|x - s| f(s) ds

    The kernel ln|x - s| has a *weak logarithmic singularity*, which is
    handled numerically using a product-integration quadrature scheme.

    Notes:
    ------
    - The equation is discretized over n+1 uniform nodes on [-1, 1]
      (n intervals, matching the convention used by product_trapz_weights).
    - After discretization, the continuous FIE becomes a linear system:
          (I - lam * K_w) f = g,
      where K_w is the matrix of modified quadrature weights for the kernel.
    - This solver can be used for many physical problems that lead to
      logarithmic-kernel FIEs, e.g.:
          * Fluid pressure induced normal stress on a fracture

    Parameters
    ----------
    lam : float
        Coupling coefficient lam that multiplies the integral term.
        (e.g. in fluid induced normal stress case: lam = (4/pi)*(k_n/E'))

    RHS : callable
        Function g(x) representing the known forcing term.
        Must accept and return NumPy arrays.
        (e.g. fluid pressure profile)

    n : int, optional
        Number of intervals along [-1, 1] (n+1 collocation nodes).
        Default: 512.

    check_cond : bool, optional
        Whether to compute np.linalg.cond(M) and warn if ill-conditioned.
        This is an O(n^3) SVD and dominates runtime for large n (e.g.
        ~2s at n=2000 vs ~0.2s to build Kw and ~0.15s to solve). Default
        True for safety; set False once you've confirmed the operator
        is well-conditioned for your (lam, n) regime and want speed.

    Returns
    -------
    s : ndarray of shape (n+1,)
        Collocation points along the domain [-1, 1].

    sol : ndarray of shape (n+1,)
        Numerical solution f(s) to the integral equation.

    rhs : ndarray of shape (n+1,)
        Evaluated right-hand-side g(s).
    """

    # --- 1. Discretize the domain (integration variable s) ---
    s = np.linspace(-1, 1, n + 1)

    # --- 2. Build the logarithmic kernel weight matrix (cached per n) ---
    Kw = _product_trapz_weights_cached(n)

    # --- 3. Construct the discrete linear system matrix ---
    #     f(x) = g(x) + lam * int K(x,s) f(s) ds  ->  (I - lam*Kw) f = g
    M = np.eye(n + 1) - lam * Kw

    # --- 4. Check condition number for numerical stability (optional: this
    #        is an O(n^3) SVD and dominates runtime at large n) ---
    if check_cond:
        cond_num = np.linalg.cond(M)
        if cond_num > 1e12:
            print(f"WARNING: Ill-conditioned system matrix (cond ~ {cond_num:.2e})")

    # --- 5. Evaluate the right-hand side g(x) on the grid ---
    rhs = RHS(s)

    # --- 6. Solve for f(s) using direct linear algebra ---
    sol = np.linalg.solve(M, rhs)

    return s, sol, rhs


@lru_cache(maxsize=8)
def _product_trapz_weights_cached(n):
    """
    Cache the weight matrix itself, keyed on n. This is the expensive
    O(n^2) object (assembly + the psi evaluations), so caching *this*
    gives far more benefit than caching psi0/psi1 values individually.
    Rebuilding for a fixed n=1024 costs ~a few ms; repeated calls with
    the same n (e.g. sweeping over lam or RHS) become free after the first.
    """
    x = np.linspace(-1, 1, n + 1)
    return product_trapz_weights(x)


def product_trapz_weights(x):
    """
    Compute product-integration weights for integrals of the form:

        I(x_i) = ∫_{-1}^{1} f(s) ln|x_i - s| ds

    using the **product trapezoidal method** (Atkinson, 1997).

    This method corrects the ordinary trapezoidal rule by adding
    analytical terms that properly handle the weak logarithmic
    singularity at s = x_i.

    Parameters
    ----------
    x : ndarray
        1D array of collocation points (must be uniform).
        Length = n + 1 -> n intervals.

    Returns
    -------
    weights : ndarray of shape (n+1, n+1)
        Weight matrix such that:

            I(x_i) ~= sum_j weights[i, j] * f(x_j)

        Each row corresponds to the integral evaluated at x_i.
    """

    # --- Number of intervals and spacing ---
    n = len(x) - 1
    h = (x[-1] - x[0]) / n

    # --- Every offset k = i - j that can occur, computed exactly once ---
    #     k ranges over -n .. n  (2n+1 unique values)
    k_unique = np.arange(-n, n + 1)
    # --- Evaluate auxiliary digamma-like correction functions ---
    #     psi0(k) and psi1(k) are analytical functions that arise in
    #     the derivation of the product trapezoidal weights for ln|x-s|.
    psi0_lut = psi0(k_unique)  # index with k + n
    psi1_lut = psi1(k_unique)

    log_h_half = h * np.log(h) / 2

    weights = np.zeros((n + 1, n + 1))
    i_all = np.arange(n + 1)

    # Accumulate contribution of each interval [x_{m-1}, x_m] into every
    # collocation row i. This is the correct product-trapezoidal
    # assembly: each of the n intervals contributes an "alpha" weight to
    # its left node and a "beta" weight to its right node, for every x_i.
    for m in range(1, n + 1):
        k = i_all - m + 1
        idx = k + n  # index into the deduplicated LUT
        p0 = psi0_lut[idx]
        p1 = psi1_lut[idx]

        # --- Compute the alpha (alpha) and β (beta) coefficients ---
        #     These modify the basic trapezoidal weights near the singularity.
        #     See Atkinson (1967) for full derivation.
        alpha_m = log_h_half + h * (p0 - p1)  # weight contributed to x_{m-1}
        beta_m = log_h_half + h * p1  # weight contributed to x_m

        # --- Assemble the full (n+1)×(n+1) weight matrix ---
        weights[:, m - 1] += alpha_m
        weights[:, m] += beta_m

    return weights


def psi0(k):
    """
    Approximation of the digamma-like function:
        f(k) = (1-k) * log|1-k| + k * log|k| - 1    for |k| < 100, k != 0,1
        f(k) = log|k| - 1/(2k) - 1/(6k^2) - 1/(15k^3) for |k| >= 100
        f(k) = -1   for k = 0 or k = 1

    Parameters
    ----------
    k : array_like or float
        Input value(s).

    Returns
    -------
    out : ndarray or float
        Computed values.
    """
    k = np.asarray(k, dtype=int)
    scalar_input = k.ndim == 0
    k = np.atleast_1d(k)
    out = np.empty_like(k, dtype=float)

    # Singularities: f(0) = f(1) = -1
    singular = (k == 0) | (k == 1)
    out[singular] = -1.0

    # Moderate values: |k| < 100 and not singular
    mask_mid = (~singular) & (np.abs(k) < 100)
    if np.any(mask_mid):
        km = k[mask_mid]
        out[mask_mid] = (1 - km) * np.log(np.abs(1 - km)) + km * np.log(np.abs(km)) - 1

    # Large |k|: asymptotic expansion
    mask_large = (~singular) & (np.abs(k) >= 100)
    if np.any(mask_large):
        kl = k[mask_large]
        out[mask_large] = (
            np.log(np.abs(kl)) - 1 / (2 * kl) - 1 / (6 * kl**2) - 1 / (15 * kl**3)
        )

    # Return scalar if input was scalar
    return out.item() if scalar_input else out


def psi1(k):
    """
    Approximation of a digamma-like companion function:
        f(k) = 0.5*k^2*log|k| + 0.5*(1-k^2)*log|k-1| - k/2 - 1/4     for |k| < 100, k != 0,1
        f(0) = -0.25
        f(1) = -0.75
        f(k) ~= 0.5*k^2*log|k| - k/2 - 1/4 + asymptotic corrections   for |k| >= 100

    Parameters
    ----------
    k : array_like or float
        Input value(s).

    Returns
    -------
    out : ndarray or float
        Computed values.
    """
    k = np.asarray(k, dtype=int)
    scalar_input = k.ndim == 0
    k = np.atleast_1d(k)
    out = np.empty_like(k, dtype=float)

    # Special cases
    mask0 = k == 0
    mask1 = k == 1
    out[mask0] = -0.25
    out[mask1] = -0.75

    # Moderate values: |k| < 100 and not 0 or 1
    mask_mid = (~mask0 & ~mask1) & (np.abs(k) < 100)
    if np.any(mask_mid):
        km = k[mask_mid]
        out[mask_mid] = (
            0.5 * km**2 * np.log(np.abs(km))
            + 0.5 * (1 - km**2) * np.log(np.abs(km - 1))
            - km / 2
            - 0.25
        )

    # Large |k|: asymptotic expansion
    mask_large = (~mask0 & ~mask1) & (np.abs(k) >= 100)
    if np.any(mask_large):
        kl = k[mask_large]
        # Leading behavior + expansion terms
        out[mask_large] = (
            0.5 * np.log(np.abs(kl)) - 1 / (3 * kl) - 1 / (8 * kl**2) - 1 / (15 * kl**3)
        )

    # Return scalar if input was scalar
    return out.item() if scalar_input else out
