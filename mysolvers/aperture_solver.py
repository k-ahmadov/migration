import numpy as np
from scipy.linalg import solve_banded
from scipy.sparse import csc_array, diags
from scipy.sparse.linalg import factorized, spsolve
from scipy.stats import hmean


def solve_nonlinear_diffusion_n3_constant_flux(
    Nx: int, Nt: int, ui_hat: float, T_hat: float
) -> np.ndarray:
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
    ui_hat : float
        Uniform Initial condition (dimensionless) for û at t̂ = 0.
    T_hat : float
        Final dimensionless simulation time.

    Returns
    -------
    u_hat : ndarray of shape (Nt, Nx)
        Dimensionless solution û(x̂, t̂) at each time step and spatial cell.

    Notes
    -----
    - The scheme uses harmonic means for flux interpolation at cell interfaces.
    - Fully implicit (backward Euler) in time.
    """
    dx_hat = 1.0 / Nx
    dt_hat = T_hat / Nt
    gamma = dt_hat / dx_hat**2

    u_hat = np.zeros((Nt, Nx))
    u_hat[0, :] = ui_hat

    for k in range(Nt - 1):
        u_k = u_hat[k, :]

        # harmonic mean of u³ between adjacent cells -> face diffusivities
        alpha = hmean(np.array([u_k[:-1] ** 3, u_k[1:] ** 3]), axis=0)

        # tridiagonal system from implicit FVM discretization of ∂/∂x̂(û³ ∂û/∂x̂)
        # (I - gamma * Laplacian(alpha))
        off_diag = -gamma * alpha
        main_diag = np.ones(Nx)
        main_diag[:-1] += gamma * alpha
        main_diag[1:] += gamma * alpha

        A = csc_array(diags([off_diag, main_diag, off_diag], offsets=[-1, 0, 1]))  # type: ignore

        # RHS: constant-flux Neumann condition at left boundary
        b = u_k.copy()
        b[0] += gamma * dx_hat

        u_hat[k + 1, :] = spsolve(A, b)

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


def solve_linear_radial_diffusion(
    num_nodes: int, num_steps: int, w_initial: float, r_b: float, t_final: float
):
    """
    Solve the dimensionless axisymmetric linear diffusion
    equation using Control Volume method

    Equation (dimensionless form):
        ∂w/∂t = 1/r ∂/∂r (r ∂w/∂r)

    Dimensionless variables:
        r -> r / L
        w -> w / W*,     where W* = w / w_b (w_b - aperture at borehole)
        t -> t / T*,     where T* = L² / D

    Parameters:
    ----------
        num_nodes: int
            Number of spatial nodes
        num_steps: int
            Number of time steps
        r_b: float
            Dimensionless borehole radius (r_b -> r_b / L)
        w_initial: float
            Uniform initial condition
        t_final : float
            Final dimensionless simulation time.

    Returns
    -------
    w : ndarray of shape (num_steps+1, num_nodes)
        Dimensionless solution û(x̂, t̂) at each time step and spatial cell.
    """
    dr = (1 - r_b) / (num_nodes - 1)
    # building the grid
    # nodes (vertices)
    r_nodes = r_b + (np.arange(1, num_nodes + 1) - 1) * dr
    # cell centers (faces)
    r_faces = r_b + (np.arange(1, num_nodes) - 1 / 2) * dr
    # cell volumes
    V = np.zeros(num_nodes)
    V[0] = 0.5 * (r_faces[0] ** 2 - r_nodes[0] ** 2)
    V[1:-1] = 0.5 * (r_faces[1:] ** 2 - r_faces[:-1] ** 2)
    V[-1] = 0.5 * (r_nodes[-1] ** 2 - r_faces[-1] ** 2)

    # coefficients
    a = np.zeros(num_nodes)  # coef. of w_{i-1}
    b = np.zeros(num_nodes)  # coef. of w_{i}
    c = np.zeros(num_nodes)  # coef. of w_{i+1}

    dt = t_final / (num_steps - 1)
    alpha = dt / dr

    # interior nodes (all nodes except i=0 and i=N)
    for i in range(1, num_nodes - 1):
        rm, rp = r_faces[i - 1], r_faces[i]
        gamma = alpha / V[i]
        a[i] = -gamma * rm
        c[i] = -gamma * rp
        b[i] = 1.0 - a[i] - c[i]

    # outer node (boundary conditions)
    # constant aperture at the injection point
    a[0], b[0], c[0] = 0.0, 1.0, 0.0
    # no flux at i=N
    rm = r_faces[-1]
    a[-1] = (-alpha / V[-1]) * rm
    c[-1] = 0.0
    b[-1] = 1 - a[-1]

    # banded matrix that contains only 3 diagonals
    # ab[u + i - j, j] == A[i, j], u is number of non-zero upper diags
    ab = np.zeros((3, num_nodes))
    # superdiagonal
    ab[0, 1:] = c[:-1]
    # main diagonal
    ab[1, :] = b
    # subdiagonal
    ab[2, :-1] = a[1:]

    w = np.zeros((num_steps, num_nodes))
    # initial condition
    w[0, :] = w_initial

    for n in range(num_steps - 1):
        d_rhs = w[n, :].copy()
        # dirichlet bc at i=1
        d_rhs[0] = 1
        w[n + 1, :] = solve_banded((1, 1), ab, d_rhs)

    return w


if __name__=="__main__":
    import matplotlib.pyplot as plt
    num_nodes = 100
    num_steps = 100
    # all input parameters are dimensionless
    w_i = 1e-5 / 2e-5
    L = 100
    r_b = 0.1 / L
    r = np.linspace(r_b, 1, num_nodes)
    D = 0.01
    t_final = 0.1

    w_tx = solve_linear_radial_diffusion(num_nodes, num_steps, w_i, r_b, t_final)

    dwdr_outer = (w_tx[-1, -1] - w_tx[-1, -2]) / (r[-1] - r[-2])
    print(f"dw/dr at outer boundary (should be ~0): {dwdr_outer:.2e}")

    plt.figure()
    for n in [1, 10, 50, -1]:
        plt.plot(r, w_tx[n])
    plt.show()
