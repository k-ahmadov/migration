import numpy as np
from numpy.typing import NDArray
from scipy.special import erfc, exp1


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


def solve_linear_radial_diffusion(
    theta_inf: float, zeta_b: float, zeta: NDArray[np.float64]
) -> NDArray[np.float64]:
    """
    Solution of axisymmetric linear diffusion equation in similarity
    form semi-infinite and constant aperture boundary condition with
    uniform initial aperture given by theta_inf
    """
    exp_integral_part = 1 - exp1(zeta**2) / exp1(zeta_b**2)
    theta = 1 + (theta_inf - 1) * exp_integral_part
    return theta
