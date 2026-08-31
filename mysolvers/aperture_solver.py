from typing import Callable

import numpy as np
from scipy.linalg import solve_banded
from scipy.stats import hmean


def _face_conductivity(
    w: np.ndarray, k_func: Callable[[np.ndarray], np.ndarray]
) -> np.ndarray:
    """Harmonic mean of k(w) evaluated at adjacent nodes -> face values."""
    kw = k_func(w)
    return hmean(np.array([kw[:-1], kw[1:]]), axis=0)


def _assemble(
    num_nodes: int,
    w: np.ndarray,
    k_func: Callable[[np.ndarray], np.ndarray],
    dt: float,
    t_new: float,
    left_bc_constant_rate: bool = True,
):
    dx = 1.0 / (num_nodes - 1)
    gamma = dt / dx**2

    # tridiagonal coefficients
    a = np.zeros(num_nodes)  # coef. of w_{i-1}
    b = np.zeros(num_nodes)  # coef. of w_{i}
    c = np.zeros(num_nodes)  # coef. of w_{i+1}

    k_face = _face_conductivity(w, k_func)

    for i in range(1, num_nodes - 1):
        km, kp = k_face[i - 1], k_face[i]
        a[i] = -gamma * km
        c[i] = -gamma * kp
        b[i] = 1 - a[i] - c[i]

    # boundary conditions (outer nodes)
    # injection rate condition
    a[0] = 0
    c[0] = -gamma * k_face[0]
    b[0] = 1 - c[0]
    # no flux at i=N
    a[-1] = -gamma * k_face[-1]
    c[-1] = 0
    b[-1] = 1 - a[-1]

    d = np.zeros(num_nodes)
    if left_bc_constant_rate:
        q = 1.0
    else:
        q = t_new
    # injection rate dependence on time only changes the coefficient d
    d[0] += dt / dx * q

    return a, b, c, d


def _to_banded(a, b, c):
    num_nodes = len(b)
    # banded matrix that contains only 3 diagonals
    # ab[u + i - j, j] == A[i, j], u is number of non-zero upper diags
    ab = np.zeros((3, num_nodes))
    # superdiagonal
    ab[0, 1:] = c[:-1]
    # main diagonal
    ab[1, :] = b
    # subdiagonal
    ab[2, :-1] = a[1:]
    return ab


def solve_diffusion(
    num_nodes: int,
    num_steps: int,
    w_initial: float,
    t_final: float,
    k_func: Callable[[np.ndarray], np.ndarray],
    left_bc_constant_rate: bool = True,
) -> np.ndarray:
    """
    Solve the nonlinear diffusion equation (n = 3)
    using the Control-Volume Method in dimensionless form.

    Equation (dimensionless form):
        ∂û/∂t̂ = ∂/∂x̂ (û³ ∂û/∂x̂)

    Dimensionless variables:
        x̂ = x / L
        û = u / U*
        t̂ = t / T*

    Parameters
    ----------
    num_notes : int
        Number of spatial nodes.
    num_steps : int
        Number of time steps.
    w_initial : float
        Uniform Initial condition (dimensionless) for û at t̂ = 0.
    t_final : float
        Final dimensionless simulation time.
    k_func : callable
        Nonlinear conductivity k(w), e.g. `lambda w: w**3`. Pass
        `lambda w: np.ones_like(w)` (or a constant array) to recover a
        linear diffusion problem.

    Returns
    -------
    w : ndarray of shape (num_steps, num_nodes)
        Dimensionless solution û(x̂, t̂) at each time step and spatial cell.

    Notes
    -----
    - The scheme uses harmonic means for flux interpolation at node intefaces.
    - Fully implicit (backward Euler) in time.
    """
    dt = t_final / (num_steps - 1)
    # tridiagonal coefficients
    w = np.zeros((num_steps, num_nodes))
    # initial condition
    w[0, :] = w_initial
    for n in range(num_steps - 1):
        w_n = w[n, :].copy()
        t_new = (n + 1) * dt
        # assemble the triag. cofficients
        a, b, c, d = _assemble(num_nodes, w_n, k_func, dt, t_new, left_bc_constant_rate)
        # build banded matrix
        ab = _to_banded(a, b, c)
        d += w_n
        w[n + 1, :] = solve_banded((1, 1), ab, d)
    return w[1:, :]


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


if __name__ == "__main__":
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
