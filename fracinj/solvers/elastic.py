# code for numerically integrating Fredholm Integral Equation of second kind
# initially made to find the induced normal stress along the fracture during fluid injection
# but can be used for an FIE with logarithmic singularity
from functools import lru_cache

import numpy as np


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
        Default: 1024.

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


# ============================================================
# Product trapezoidal rule weights for logarithmic kernel integrals
# ============================================================
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
