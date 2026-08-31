from dataclasses import dataclass

import numpy as np

type OneD = tuple[int]
type TwoD = tuple[int, int]
type Bool = np.dtype[np.bool_]
type Float64 = np.dtype[np.float64]
type Vector = np.ndarray[OneD, Float64]
type Field = np.ndarray[TwoD, Float64]

@dataclass
class Parameters:
    E: float = 0.0  # young's modulus
    L: float = 0.0  # fracture length
    k_n: float = 0.0  # normal stiffness
    mu: float = 0.0  # fluid viscosity
    nu: float = 0.0  # Poisson's ratio
    w_i: float = 0.0  # initial aperture
    q_0: float = 0.0 # injection rate
    m_q: float = 0.0 # slope of linearly increasing injection rate
    DP: float  = 0.0  # constant overpressure
    w_max: float = 0.0  # max aperture
    w_min: float = 0.0  # min aperture
