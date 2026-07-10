from dataclasses import dataclass

import numpy as np

type OneD = tuple[int]
type TwoD = tuple[int, int]
type Bool = np.dtype[np.bool_]
type Float64 = np.dtype[np.float64]
type Time = np.ndarray[OneD, Float64]  # shape: (n_t, )
type XPositions = np.ndarray[OneD, Float64]
type Field = np.ndarray[TwoD, Float64]
type HydraulicDiffusivity = float
type ParameterM = float
type ScalingExponent = float
type Prefactor = float
type CharacteristicTime = float
type DimensionlessApertureAtFront = float
type FractionalPercentage = float
type CriticalPressure = float
type DimpensionlessDistanceAtFront = float
type FrontDetectionThreshold = np.ndarray[OneD, Float64]  # shape: (n_t, )
type TimestepHasFront = np.ndarray[OneD, Bool]  # shape: (n_t, )
type FrontPositions = np.ndarray[OneD, Float64]  # shape: (n_t_has_front, )
type DimensionlessDistance = np.ndarray[OneD, Float64]
type DimensionlessAperture = np.ndarray[OneD, Float64]


@dataclass
class Parameters:
    E: float = 0.0  # young's modulus
    L: float = 0.0  # fracture length
    k_n: float = 0.0  # normal stiffness
    mu: float = 0.0  # fluid viscosity
    nu: float = 0.0  # Poisson's ratio
    w_i: float = 0.0  # initial aperture
    # optional
    q: float | None = None  # injection rate
    q_0: float | None = None  # injection rate
    DP: float | None = None  # injection rate
    w_max: float | None = None  # max aperture
    w_min: float | None = None  # min aperture

    def __post_init__(self):
        if self.q is None and self.q_0 is None and self.DP is None:
            raise ValueError(
                "At least one of injection conditions 'DP', 'q' or 'q_0' must be provided."
            )

    @property
    def flux(self) -> float:
        q = self.q or self.q_0
        if q is None:
            raise ValueError("No flux parameter set.")
        return q
