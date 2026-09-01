"""Control-volume solvers for the aperture (nonlinear) diffusion equation."""

from typing import Callable

import numpy as np
from scipy.linalg import solve_banded
from scipy.stats import hmean

KFunc = Callable[[np.ndarray], np.ndarray]


def _face_conductivity(w: np.ndarray, k_func: KFunc) -> np.ndarray:
    """Harmonic mean of ``k(w)`` at adjacent nodes -> face values."""
    kw = k_func(w)
    return hmean(np.array([kw[:-1], kw[1:]]), axis=0)


def _to_banded(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Pack tridiagonal coefficients into ``scipy.linalg.solve_banded`` form.

    ``ab[u + i - j, j] == A[i, j]`` with one upper diagonal (``u = 1``).
    """
    ab = np.zeros((3, len(b)))
    ab[0, 1:] = c[:-1]  # superdiagonal
    ab[1, :] = b  # main diagonal
    ab[2, :-1] = a[1:]  # subdiagonal
    return ab


def _assemble(
    num_nodes: int,
    w: np.ndarray,
    k_func: KFunc,
    dt: float,
    t_new: float,
    left_bc_constant_rate: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dx = 1.0 / (num_nodes - 1)
    gamma = dt / dx**2
    k_face = _face_conductivity(w, k_func)

    a = np.zeros(num_nodes)  # coef. of w_{i-1}
    b = np.zeros(num_nodes)  # coef. of w_{i}
    c = np.zeros(num_nodes)  # coef. of w_{i+1}

    a[1:-1] = -gamma * k_face[:-1]
    c[1:-1] = -gamma * k_face[1:]
    b[1:-1] = 1 - a[1:-1] - c[1:-1]

    # left node: prescribed injection rate
    c[0] = -gamma * k_face[0]
    b[0] = 1 - c[0]
    # right node: no flux
    a[-1] = -gamma * k_face[-1]
    b[-1] = 1 - a[-1]

    d = np.zeros(num_nodes)
    q = 1.0 if left_bc_constant_rate else t_new
    d[0] = dt / dx * q
    return a, b, c, d


def solve_diffusion(
    num_nodes: int,
    num_steps: int,
    w_initial: float,
    t_final: float,
    k_func: KFunc,
    left_bc_constant_rate: bool = True,
) -> np.ndarray:
    """Solve ``d w/dt = d/dx (k(w) dw/dx)`` in dimensionless form (implicit Euler).

    Pass ``k_func = lambda w: w**3`` for the n=3 nonlinear case, or
    ``lambda w: np.ones_like(w)`` for linear diffusion.

    Returns
    -------
    w : ndarray, shape (num_steps - 1, num_nodes)
        Solution at each time step after the initial condition.
    """
    dt = t_final / (num_steps - 1)
    w = np.zeros((num_steps, num_nodes))
    w[0, :] = w_initial

    for n in range(num_steps - 1):
        w_n = w[n, :]
        a, b, c, d = _assemble(
            num_nodes, w_n, k_func, dt, (n + 1) * dt, left_bc_constant_rate
        )
        w[n + 1, :] = solve_banded((1, 1), _to_banded(a, b, c), d + w_n)
    return w[1:, :]


def solve_linear_radial_diffusion(
    num_nodes: int, num_steps: int, w_initial: float, r_b: float, t_final: float
) -> np.ndarray:
    """Axisymmetric linear diffusion ``dw/dt = (1/r) d/dr (r dw/dr)``.

    Dirichlet (``w = 1``) at the borehole ``r_b``, no flux at ``r = 1``.

    Returns
    -------
    w : ndarray, shape (num_steps, num_nodes)
    """
    dr = (1 - r_b) / (num_nodes - 1)
    r_nodes = r_b + np.arange(num_nodes) * dr
    r_faces = r_b + (np.arange(1, num_nodes) - 0.5) * dr

    V = np.zeros(num_nodes)
    V[0] = 0.5 * (r_faces[0] ** 2 - r_nodes[0] ** 2)
    V[1:-1] = 0.5 * (r_faces[1:] ** 2 - r_faces[:-1] ** 2)
    V[-1] = 0.5 * (r_nodes[-1] ** 2 - r_faces[-1] ** 2)

    dt = t_final / (num_steps - 1)
    alpha = dt / dr

    a = np.zeros(num_nodes)
    b = np.zeros(num_nodes)
    c = np.zeros(num_nodes)
    gamma = alpha / V[1:-1]
    a[1:-1] = -gamma * r_faces[:-1]
    c[1:-1] = -gamma * r_faces[1:]
    b[1:-1] = 1.0 - a[1:-1] - c[1:-1]

    b[0] = 1.0  # Dirichlet at the borehole
    a[-1] = (-alpha / V[-1]) * r_faces[-1]  # no flux at r = 1
    b[-1] = 1 - a[-1]

    ab = _to_banded(a, b, c)
    w = np.zeros((num_steps, num_nodes))
    w[0, :] = w_initial
    for n in range(num_steps - 1):
        rhs = w[n, :].copy()
        rhs[0] = 1.0
        w[n + 1, :] = solve_banded((1, 1), ab, rhs)
    return w
