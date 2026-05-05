import numpy as np
from numpy.typing import NDArray
from scipy.special import erfc


def solve_linear_diffusion_const_flux(zeta: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Solution of 1D linear diffusion equation in similarity form
    semi-infinite and flux boundary conditions
    zero initial condition
    """
    part1: NDArray[np.float64] = np.sqrt(4 / np.pi) * np.exp(-(zeta**2) / 4)
    part2 = zeta * erfc(zeta / 2)
    theta: NDArray[np.float64] = part1 - part2
    return theta
