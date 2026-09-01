"""Closed-form similarity solutions of the linear diffusion equation."""

import numpy as np
from numpy.typing import NDArray
from scipy.special import erfc, exp1


def solve_linear_diffusion_const_flux(
    zeta: NDArray[np.float64],
) -> NDArray[np.float64]:
    """1D linear diffusion, semi-infinite, constant-flux BC, zero initial aperture."""
    return np.sqrt(4 / np.pi) * np.exp(-(zeta**2) / 4) - zeta * erfc(zeta / 2)


def solve_linear_radial_diffusion(
    theta_inf: float, zeta_b: float, zeta: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Axisymmetric linear diffusion, constant-aperture BC, uniform initial aperture."""
    return 1 + (theta_inf - 1) * (1 - exp1(zeta**2) / exp1(zeta_b**2))
