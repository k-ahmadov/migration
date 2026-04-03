# code for numerically integrating Fredholm Integral Equation of second kind 
# initially made to find the induced normal stress along the fracture during fluid injection
# but can be used for an FIE with logarithmic singularity
import numpy as np

def FIE_log_sing(lam, RHS, n=2**10):
    """
        Solve a general Fredholm Integral Equation (FIE) of the second kind
        with a logarithmic kernel using the product integration (Atkinson) method:

            f(x) = g(x) + λ ∫_{-1}^{1} ln|x - s| f(s) ds

        The kernel ln|x - s| has a *weak logarithmic singularity*, which is
        handled numerically using a product-integration quadrature scheme.

        Notes:
        ------
        - The equation is discretized over a uniform grid on [-1, 1].
        - After discretization, the continuous FIE becomes a linear system:
              (I - λ K_w) f = g,
          where K_w is the matrix of modified quadrature weights for the kernel.
        - This solver can be used for many physical problems that lead to
          logarithmic-kernel FIEs, e.g.:
              * Fluid pressure induced normal stress on a fracture

        Parameters
        ----------
        lam : float
            Coupling coefficient λ that multiplies the integral term.
            (e.g. in fluid induced normal stress case: λ = (4/π)*(k_n/E'))

        RHS : callable
            Function g(x) representing the known forcing term.
            Must accept and return NumPy arrays.
            (e.g. fluid pressure profile)

        n : int, optional
            Number of discretization (collocation) points along [-1, 1].
            Default: 1024 (high accuracy for smooth g(x)).

        Returns
        -------
        s : ndarray of shape (n,)
            Collocation points along the domain [-1, 1].

        sol : ndarray of shape (n,)
            Numerical solution f(s) to the integral equation.

        rhs : ndarray of shape (n,)
            Evaluated right-hand-side g(s).
    """

    # --- 1. Discretize the domain (integration variable s) ---
    s = np.linspace(-1, 1, n)

    # --- 2. Build the logarithmic kernel weight matrix ---
    #     This is where the singularity is handled by the product integration method.
    #     Kw[i, j] approximates ∫ ln|x_i - s_j| * f_j(s_j) ds
    Kw = product_trapz_weights(s)

    # --- 3. Construct the discrete linear system matrix ---
    #     The continuous equation f(x) = g(x) + λ ∫ K(x,s)f(s)ds
    #     becomes (I - λ Kw) f = g  →  f = (I - λ Kw)^(-1) g
    I = np.eye(n)
    M = I - lam * Kw

    # --- 4. Check condition number for numerical stability ---
    cond_num = np.linalg.cond(M)
    if cond_num > 1e12:
        print(f"WARNING: Ill-conditioned system matrix (cond ≈ {cond_num:.2e})")

    # --- 5. Evaluate the right-hand side g(x) on the grid ---
    rhs = RHS(s)

    # --- 6. Solve for f(s) using direct linear algebra ---
    sol = np.linalg.solve(M, rhs)

    return s, sol, rhs

# to do in future: - build psi table obtained from 
# product trapz method function below

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
            Length = n + 1 → n intervals.

        Returns
        -------
        weights : ndarray of shape (n+1, n+1)
            Weight matrix such that:

                I(x_i) ≈ Σ_j weights[i, j] * f(x_j)

            Each row corresponds to the integral evaluated at x_i.
    """

    # --- Number of intervals and spacing ---
    n = len(x) - 1
    h = (x[-1] - x[0]) / n

    # --- Index setup for offset k = i - j + 1 ---
    #     This defines how far each collocation point x_i is from each node x_j.
    i = np.arange(n + 1)             # i = [0, 1, ..., n]
    j = np.arange(1, n + 1)[:, None] # j = [1, 2, ..., n], column vector
    k = i - j + 1                    # offset matrix (shape n x n+1)

    # --- Evaluate auxiliary digamma-like correction functions ---
    #     psi0(k) and psi1(k) are analytical functions that arise in
    #     the derivation of the product trapezoidal weights for ln|x-s|.
    psi0_vals = psi0(k)
    psi1_vals = psi1(k)

    # --- Compute the α (alpha) and β (beta) coefficients ---
    #     These modify the basic trapezoidal weights near the singularity.
    #     See Atkinson (1967) for full derivation.
    alpha = h * np.log(h) / 2 + h * (psi0_vals - psi1_vals)
    beta  = h * np.log(h) / 2 + h * psi1_vals

    # --- Assemble the full (n+1)×(n+1) weight matrix ---
    weights = np.zeros((n + 1, n + 1))
    weights[0]   = alpha[0]                # weights for first collocation point
    weights[n]   = beta[-1]                # weights for last collocation point
    weights[1:n] = beta[:-1] + alpha[1:]   # weights for interior points

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
