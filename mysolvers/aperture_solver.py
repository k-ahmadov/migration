import numpy as np
from scipy.sparse import diags, csc_array
from scipy.sparse.linalg import factorized
from scipy.stats import hmean


def solve_dimless_nonlinear_diffusion_n3_constant_flux(Nx, Nt, ui_hat, T_hat):
    """
        Solve the nonlinear diffusion equation (n = 3)
        using the Finite Volume Method (FVM) in dimensionless form.

        Equation (dimensionless form):
            ∂û/∂t̂ = ∂/∂x̂ (û³ ∂û/∂x̂)

        Dimensionless variables:
            x̂ = x / L
            û = u / U*,     where U* = (L·q₀ / a)^(1/4)
            t̂ = t / T*,     where T* = L² / (a·U*³)

        Parameters
        ----------
        Nx : int
            Number of spatial cells.
        Nt : int
            Number of time steps.
        ui_hat : array_like
            Initial condition (dimensionless) for û at t̂ = 0.
        T_hat : float
            Final dimensionless simulation time.

        Returns
        -------
        u_hat : ndarray of shape (Nt, Nx)
            Dimensionless solution û(x̂, t̂) at each time step and spatial cell.

        Notes
        -----
        - The scheme uses harmonic means for flux interpolation at cell interfaces.
    """
    # mesh size
    dx_hat = 1 / Nx
    # time step
    dt_hat = T_hat / Nt
    # gamma is for convenience
    gamma = dt_hat / dx_hat**2
    # initialize the matrix of size (Nt, Nx)
    u_hat = np.zeros((Nt, Nx))
    # apply initial condition
    u_hat[0, :] = ui_hat
    # building and solving the matrix for each timestep
    for k in range(Nt - 1):
        # harmonic mean between two cells for the nonlinear term
        alpha = hmean([u_hat[k, :-1] ** 3, u_hat[k, 1:] ** 3])
        # building the matrix for implicit form
        # upper diagonal
        upper = -gamma * alpha
        # diagonal itself
        main = np.zeros(Nx)
        main[0] = 1 + gamma * alpha[0]
        main[1:-1] = 1 + gamma * (alpha[:-1] + alpha[1:])
        main[-1] = 1 + gamma * alpha[-1]
        # lower diag
        lower = -gamma * alpha
        # RHS vector
        b = u_hat[k, :].copy()
        b[0] += gamma * dx_hat
        # inverse of matrix with a method adapted for sparse matrix
        A = csc_array(diags([lower, main, upper], [-1, 0, 1]))
        solve = factorized(A)
        u_hat[k + 1, :] = solve(b)
    return u_hat


def solve_linear_diffusion_BC_constant_flux(Nx, Nt, T_hat, ui_hat=0):
    x_hat = np.linspace(0, 1, Nx)
    dx_hat = 1 / Nx
    dt_hat = T_hat / Nt
    gamma = dt_hat / dx_hat**2
    main = (1 + 2 * gamma) * np.ones(Nx)
    upper = -gamma * np.ones(Nx - 1)
    upper[0] = -2 * gamma
    lower = -gamma * np.ones(Nx - 1)
    lower[-1] = -2 * gamma
    A = csc_array(diags([lower, main, upper], [-1, 0, 1]))  # type: ignore
    solve = factorized(csc_array(A))
    u_hat = np.zeros((Nt, Nx))
    u_hat[0, :] = ui_hat
    for k in range(Nt - 1):
        b = u_hat[k, :].copy()
        b[0] += 2 * gamma * dx_hat
        u_hat[k + 1, :] = solve(b)
    return x_hat, u_hat
