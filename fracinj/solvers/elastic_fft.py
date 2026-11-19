"""
Normal stress change on a fluid-pressurized fracture (elastic joint between two
elastic half-planes, plane strain, normal loading only, no shear traction).

PROBLEM
-------
A fracture has normal stiffness k_n and sits between two elastic half-planes
with plane-strain modulus E'. A fluid pressure change dp(x) acts in the fracture.

    Terzaghi + linear joint law:   dsigma_n - dp = -k_n v          (v = opening)
    Elasticity (both walls):       v(x) = -4/(pi E') Int dsigma_n(s) ln|x-s| ds

Eliminating v gives an integral equation for the total normal stress change:

    dsigma_n(x) - 4 k_n/(pi E') Int dsigma_n(s) ln|x-s| ds = dp(x)

NONDIMENSIONALIZATION
---------------------
    X = x/L_p,  Pi = dp/P,  Sigma = dsigma_n/P,  V = k_n v/P
    eps = L_k/L_p = E' / (4 k_n L_p)    (L_k = E'/(4 k_n): elasto-joint length,
                                         L_p = length scale of the pressure pulse)

    eps*Sigma(X) - (1/pi) Int Sigma(S) ln|X-S| dS = eps*Pi(X),     V = Pi - Sigma

SOLUTION (Fourier transform, k = wavenumber in units of 1/L_p)
-------------------------------------------------------------
    Sigma_hat = eps|k| / (1 + eps|k|) * Pi_hat
    V_hat     =    1   / (1 + eps|k|) * Pi_hat

This is exact for any eps. For eps << 1, Sigma ~ eps * Hilbert[dPi/dX] and V ~ Pi.
The net load is zero: Int Sigma dX = 0.

INPUT / OUTPUT
--------------
Input : Pi(X) on a uniform grid, decaying to ~0 at both ends (an isolated pulse).
Output: Sigma(X) and V(X) on the same grid.
"""

import warnings

import numpy as np


def solve_elastic_joint(X: np.ndarray, Pi: np.ndarray, eps: float, pad=4):
    """
    Total normal stress change Sigma and opening V for a pressure profile Pi.

    X   : uniform grid in units of L_p (increasing)
    Pi  : dimensionless pressure change on that grid (must decay to ~0 at both ends)
    eps : L_k / L_p = E' / (4 k_n L_p)
    pad : the FFT box is at least `pad` times the length of the data
    """

    X = np.asarray(X, dtype=float)
    Pi = np.asarray(Pi, dtype=float)
    if X.ndim != 1 or X.shape != Pi.shape:
        raise ValueError("X and Pi must be 1-D arrays of the same length")
    if np.any(np.diff(X) <= 0):
        raise ValueError("X must be strictly increasing")
    # Warn if the pulse is cut off at the domain edges
    if max(abs(Pi[0]), abs(Pi[-1])) > 1e-3 * np.max(np.abs(Pi)):
        warnings.warn("Pi has not decayed at the domain ends; extend the domain", stacklevel=2)

    n = len(X)           # number of grid points
    h = X[1] - X[0]      # grid spacing

    # Zero-padded FFT length: long enough for the data AND for the kernel width
    # (about eps), rounded up to a power of 2 for speed
    n_needed = max(pad * n, 50 * eps / h)
    nfft = int(2 ** np.ceil(np.log2(n_needed)))

    # Angular wavenumbers matching the FFT output
    k = 2 * np.pi * np.fft.rfftfreq(nfft, d=h)

    # Transform Pi (numpy appends the zeros up to nfft)
    Pi_hat = np.fft.rfft(Pi, nfft)

    # Multiply by the exact solution operators, then transform back
    Sigma_hat = eps * k / (1 + eps * k) * Pi_hat
    V_hat = Pi_hat / (1 + eps * k)

    # Keep only the first n points, which is the original (unpadded) domain
    Sigma = np.fft.irfft(Sigma_hat, nfft)[:n]
    V = np.fft.irfft(V_hat, nfft)[:n]
    return Sigma, V
