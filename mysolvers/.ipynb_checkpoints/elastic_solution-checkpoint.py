# code for numerically integrating Fredholm Integral Equation of second kind 
# to find the induced normal stress along the fracture during fluid injection
import numpy as np

def calc_sn(lam, params, C, pressure, n=2**10):
    s = np.linspace(-params["L"], params["L"], n)
    Kw = product_trapz_weights(s)
    I = np.eye(n)
    M = I - (4/np.pi) * lam * Kw
    # condition check
    cond_num = np.linalg.cond(M)
    if cond_num > 1e12:
        print("WARNING: ill-conditioned matrix:", cond_num)
    rhs = pressure(s, params)
    sn = np.linalg.solve(M, rhs-C) 
    return s, sn, rhs

def calc_sn_hat(lam, params, pressure, n=2**10):
    s = np.linspace(-1, 1, n)
    Kw = product_trapz_weights(s)
    I = np.eye(n)
    M = I - (4/np.pi) * lam * Kw
    # condition check
    cond_num = np.linalg.cond(M)
    if cond_num > 1e12:
        print("WARNING: ill-conditioned matrix:", cond_num)
    rhs = pressure(s, params)
    sn = np.linalg.solve(M, rhs) 
    return s, sn, rhs
 
def product_trapz_weights(x):
    # Parameters
    n = len(x) - 1
    h = (x[-1] - x[0]) / n
    # Build index differences: k = i - j + 1
    i = np.arange(n + 1)            # shape (n+1,)
    j = np.arange(1, n + 1)[:, None] # shape (n,1)
    k = i - j + 1                   # shape (n,n+1), broadcasting takes care
    # build psi0 and psi1 grid
    psi0_vals = psi0(k)
    psi1_vals = psi1(k)
    # Compute alpha and beta directly (vectorized)
    alpha = h * np.log(h) / 2 + h * (psi0_vals - psi1_vals)
    beta  = h * np.log(h) / 2 + h * psi1_vals
    # Assemble weights
    weights = np.zeros((n + 1, n + 1))
    weights[0]   = alpha[0]            # first row
    weights[n]   = beta[-1]            # last row
    weights[1:n] = beta[:-1] + alpha[1:]  # middle rows
    return weights

def psi0(k):
    """
    Approximation of the digamma-like function:
        f(k) = (1-k) * log|1-k| + k * log|k| - 1    for |k| < 100, k ≠ 0,1
        f(k) = log|k| - 1/(2k) - 1/(6k^2) - 1/(15k^3) for |k| ≥ 100
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
            np.log(np.abs(kl))
            - 1 / (2 * kl)
            - 1 / (6 * kl**2)
            - 1 / (15 * kl**3)
        )

    # Return scalar if input was scalar
    return out.item() if np.isscalar(k) else out

def psi1(k):
    """
    Approximation of a digamma-like companion function:
        f(k) = 0.5*k^2*log|k| + 0.5*(1-k^2)*log|k-1| - k/2 - 1/4     for |k| < 100, k ≠ 0,1
        f(0) = -0.25
        f(1) = -0.75
        f(k) ≈ 0.5*k^2*log|k| - k/2 - 1/4 + asymptotic corrections   for |k| ≥ 100

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
    out = np.empty_like(k, dtype=float)

    # Special cases
    mask0 = (k == 0)
    mask1 = (k == 1)
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
            0.5 * np.log(np.abs(kl))
            - 1 / (3 * kl)
            - 1 / (8 * kl**2)
            - 1 / (15 * kl**3)
        )

    # Return scalar if input was scalar
    return out.item() if np.isscalar(k) else out