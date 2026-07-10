from dataclasses import dataclass

import numpy as np

from mypackages import file_io, physics
from mypackages.types import Float64, OneD, Time
from mysolvers.exact_solutions import solve_linear_diffusion_const_flux
from mysolvers.similarity_solutions import solve_neumann_n3


def find_p_inj(run: file_io.RunData):
    _, p = file_io.sort_fields(run.x_sc, run.p)
    p_inj = p[:, 0]
    return p_inj


@dataclass
class PInjResults:
    p_inj_num: np.ndarray[OneD, Float64]
    A_ana: float
    α_ana: float

    def p_inj_analytical(self, time: Time):
        return self.A_ana * time**self.α_ana


RIGID_APERTURE_EXPONENT: float = 0.5


def analyze_rigid(run: file_io.RunData):

    p_inj = find_p_inj(run)

    θ = solve_linear_diffusion_const_flux(np.linspace(0, 10, 200))
    prefactor = (
        θ[0]
        * run.params.flux
        * run.params.k_n
        / (physics.diffusivity(run.params)) ** (RIGID_APERTURE_EXPONENT)
    )

    return PInjResults(p_inj_num=p_inj, A_ana=prefactor, α_ana=RIGID_APERTURE_EXPONENT)


SOFT_APERTURE_EXPONENT: float = 0.2


def analyze_soft(run: file_io.RunData):

    p_inj = find_p_inj(run)
    θ = solve_neumann_n3(run.params.w_i / run.w[-1][0])[1]
    prefactor = (
        θ[0]
        * run.params.k_n
        * (run.params.flux**2 / physics.parameter_M(run.params)) ** (0.2)
    )

    return PInjResults(p_inj_num=p_inj, A_ana=prefactor, α_ana=SOFT_APERTURE_EXPONENT)


# def local_exponent(x, y):
#     log_x = np.log(x)
#     log_y = np.log(y)
#     slope = np.gradient(log_y, log_x)
#     return slope
