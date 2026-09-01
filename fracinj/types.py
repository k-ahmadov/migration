"""Shared array-type aliases and the parameter container."""

from dataclasses import dataclass, field, fields

import numpy as np
from numpy.typing import NDArray

# Semantic aliases -- 1D arrays (a coordinate or time series) and 2D fields
# indexed as ``field[time, space]``.
Vector = NDArray[np.float64]
Field = NDArray[np.float64]
BoolVector = NDArray[np.bool_]


def _p(unit: str, description: str, default: float = 0.0):
    """Declare a Parameters field carrying unit/description metadata."""
    return field(default=default, metadata={"unit": unit, "description": description})


@dataclass
class Parameters:
    """Physical parameters plus discretisation settings for a run.

    Only one injection driver (``q_0``, ``m_q`` or ``DP``) is set for a
    given run; the others stay at 0.
    """

    # --- material / geometry ------------------------------------------------
    E: float = _p("Pa", "Young's modulus")
    L: float = _p("m", "Fracture length")
    k_n: float = _p("Pa/m", "Normal stiffness")
    mu: float = _p("Pa.s", "Fluid viscosity")
    nu: float = _p("-", "Poisson's ratio")
    w_i: float = _p("m", "Initial aperture")

    # --- injection driver (regime dependent) ------------------------------
    q_0: float = _p("m^2/s", "Constant injection rate")
    m_q: float = _p("m^2/s^2", "Injection-rate ramp slope")
    DP: float = _p("Pa", "Constant overpressure")

    # --- optional aperture bounds ---------------------------------------
    w_max: float = _p("m", "Maximum aperture")
    w_min: float = _p("m", "Minimum aperture")

    # --- discretisation -------------------------------------------------
    T: float = _p("s", "Simulated duration")
    Nx_p: float = _p("-", "FVM spatial cells")
    Nx_sn: float = _p("-", "Elastic-solution spatial cells")
    Nt: float = _p("-", "Time steps")

    def unit(self, name: str) -> str:
        return self.__dataclass_fields__[name].metadata["unit"]

    def description(self, name: str) -> str:
        return self.__dataclass_fields__[name].metadata["description"]


def parameter_names() -> set[str]:
    return {f.name for f in fields(Parameters)}
