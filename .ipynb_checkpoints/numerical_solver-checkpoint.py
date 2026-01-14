import numpy as np
from scipy.sparse import diags, csc_array
from scipy.sparse.linalg import factorized
from scipy.stats import hmean

def solve_fdm_diffusion_flux(Nx, Nt, T_hat, ui_hat=0):
    x_hat = np.linspace(0, 1, Nx)
    dx_hat = 1 / Nx
    dt_hat = T_hat / Nt
    gamma = dt_hat / dx_hat ** 2
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

def solve_fvm_pme_flux(Nx, Nt, ui_hat, T_hat):
    dx_hat = 1 / Nx
    dt_hat = T_hat / Nt
    gamma = dt_hat / dx_hat ** 2
    u_hat = np.zeros((Nt, Nx))
    u_hat[0, :] = ui_hat
    for k in range(Nt - 1):
        alpha = hmean([u_hat[k, :-1] ** 3, u_hat[k, 1:] ** 3])
        upper = -gamma * alpha
        main = np.zeros(Nx)
        main[0] = 1 + gamma * alpha[0]
        main[1:-1] = 1 + gamma * (alpha[:-1] + alpha[1:])
        main[-1] = 1 + gamma * alpha[-1]
        lower = -gamma * alpha
        b = u_hat[k, :].copy()
        b[0] += gamma * dx_hat
        A = csc_array(diags([lower, main, upper], [-1, 0, 1]))  # type: ignore
        solve = factorized(A)
        u_hat[k + 1, :] = solve(b)
    return u_hat